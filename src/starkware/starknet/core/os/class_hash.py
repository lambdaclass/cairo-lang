import contextlib
import dataclasses
import itertools
import json
import os
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import cachetools
import cairo_rs_py

from starkware.cairo.common.cairo_function_runner import CairoFunctionRunner
from starkware.cairo.common.structs import CairoStructFactory, CairoStructProxy
from starkware.cairo.lang.builtins.hash.hash_builtin_runner import HashBuiltinRunner
from starkware.cairo.lang.cairo_constants import DEFAULT_PRIME
from starkware.cairo.lang.compiler.ast.cairo_types import add_backward_compatibility_space
from starkware.cairo.lang.compiler.cairo_compile import compile_cairo_files
from starkware.cairo.lang.compiler.identifier_definition import ConstDefinition
from starkware.cairo.lang.compiler.identifier_manager import IdentifierManager
from starkware.cairo.lang.compiler.program import Program
from starkware.cairo.lang.compiler.scoped_name import ScopedName
from starkware.cairo.lang.vm.crypto import pedersen_hash
from starkware.python.utils import from_bytes
from starkware.starknet.public.abi import starknet_keccak
from starkware.starknet.services.api.contract_class import ContractClass, EntryPointType
# Added Imports
from starkware.cairo.lang.vm.relocatable import MaybeRelocatable, RelocatableValue
from starkware.cairo.lang.vm.vm_exceptions import SecurityError, VmException
from starkware.python.utils import safe_zip

CAIRO_FILE = os.path.join(os.path.dirname(__file__), "contracts.cairo")

class_hash_cache_ctx_var: ContextVar[Optional[cachetools.LRUCache]] = ContextVar(
    "class_hash_cache", default=None
)


@contextlib.contextmanager
def set_class_hash_cache(cache: cachetools.LRUCache):
    """
    Sets a cache to be used by compute_class_hash().
    """
    assert class_hash_cache_ctx_var.get() is None, "Cannot replace an existing class_hash_cache."

    token = class_hash_cache_ctx_var.set(cache)
    try:
        yield
    finally:
        class_hash_cache_ctx_var.reset(token)


@lru_cache()
def load_program() -> Program:
    return compile_cairo_files(
        [CAIRO_FILE],
        prime=DEFAULT_PRIME,
        main_scope=ScopedName.from_string("starkware.starknet.core.os.contracts"),
    )


def compute_class_hash(
    contract_class: ContractClass, hash_func: Callable[[int, int], int] = pedersen_hash
) -> int:
    cache = class_hash_cache_ctx_var.get()
    if cache is None:
        return compute_class_hash_inner(contract_class=contract_class, hash_func=hash_func)

    contract_class_bytes = contract_class.dumps(sort_keys=True).encode()
    key = (starknet_keccak(data=contract_class_bytes), hash_func)

    if key not in cache:
        cache[key] = compute_class_hash_inner(contract_class=contract_class, hash_func=hash_func)

    return cache[key]


def compute_class_hash_inner(
    contract_class: ContractClass, hash_func: Callable[[int, int], int]
) -> int:
    program = load_program()
    contract_class_struct = get_contract_class_struct(
        identifiers=program.identifiers, contract_class=contract_class
    )

    runner = cairo_rs_py.CairoRunner(program=program.dumps(), entrypoint=None, layout="all", proof_mode=False)
    runner.initialize_function_runner()
    hash_ptr = runner.add_additional_hash_builtin()


    run_function_runner(
        runner,
        program,
        "starkware.starknet.core.os.contracts.class_hash",
        hash_ptr=hash_ptr,
        contract_class=contract_class_struct,
        use_full_name=True,
        verify_secure=False,
    )
    _, class_hash = runner.get_return_values(2)
    return class_hash


def compute_hinted_class_hash(contract_class: ContractClass) -> int:
    """
    Computes the hash of the contract class, including hints.
    """
    program_without_debug_info = dataclasses.replace(contract_class.program, debug_info=None)

    # If compiler_version is not present, this was compiled with a compiler before version 0.10.0.
    # Use "(a : felt)" syntax instead of "(a: felt)" so that the class hash will be the same.
    with add_backward_compatibility_space(contract_class.program.compiler_version is None):
        dumped_program = program_without_debug_info.dump()

    if len(dumped_program["attributes"]) == 0:
        # Remove attributes field from raw dictionary, for hash backward compatibility of
        # contracts deployed prior to adding this feature.
        del dumped_program["attributes"]
    else:
        # Remove accessible_scopes and flow_tracking_data fields from raw dictionary, for hash
        # backward compatibility of contracts deployed prior to adding this feature.
        for attr in dumped_program["attributes"]:
            if len(attr["accessible_scopes"]) == 0:
                del attr["accessible_scopes"]
            if attr["flow_tracking_data"] is None:
                del attr["flow_tracking_data"]

    input_to_hash = dict(program=dumped_program, abi=contract_class.abi)
    return starknet_keccak(data=json.dumps(input_to_hash, sort_keys=True).encode())


def get_contract_entry_points(
    structs: CairoStructProxy,
    contract_class: ContractClass,
    entry_point_type: EntryPointType,
) -> List[CairoStructProxy]:
    # Check validity of entry points.
    program_length = len(contract_class.program.data)
    entry_points = contract_class.entry_points_by_type[entry_point_type]
    for entry_point in entry_points:
        assert (
            0 <= entry_point.offset < program_length
        ), f"Invalid entry point offset {entry_point.offset}, len(program_data)={program_length}."

    return [
        structs.ContractEntryPoint(selector=entry_point.selector, offset=entry_point.offset)
        for entry_point in entry_points
    ]


def get_contract_class_struct(
    identifiers: IdentifierManager, contract_class: ContractClass
) -> CairoStructProxy:
    """
    Returns the serialization of a contract as a list of field elements.
    """
    structs = CairoStructFactory(
        identifiers=identifiers,
        additional_imports=[
            "starkware.starknet.core.os.contracts.ContractClass",
            "starkware.starknet.core.os.contracts.ContractEntryPoint",
        ],
    ).structs

    API_VERSION_IDENT = identifiers.get_by_full_name(
        ScopedName.from_string("starkware.starknet.core.os.contracts.API_VERSION")
    )
    assert isinstance(API_VERSION_IDENT, ConstDefinition)

    external_functions, l1_handlers, constructors = (
        get_contract_entry_points(
            structs=structs,
            contract_class=contract_class,
            entry_point_type=entry_point_type,
        )
        for entry_point_type in (
            EntryPointType.EXTERNAL,
            EntryPointType.L1_HANDLER,
            EntryPointType.CONSTRUCTOR,
        )
    )
    flat_external_functions, flat_l1_handlers, flat_constructors = (
        list(itertools.chain.from_iterable(entry_points))
        for entry_points in (external_functions, l1_handlers, constructors)
    )

    builtin_list = contract_class.program.builtins
    return structs.ContractClass(
        api_version=API_VERSION_IDENT.value,
        n_external_functions=len(external_functions),
        external_functions=flat_external_functions,
        n_l1_handlers=len(l1_handlers),
        l1_handlers=flat_l1_handlers,
        n_constructors=len(constructors),
        constructors=flat_constructors,
        n_builtins=len(builtin_list),
        builtin_list=[from_bytes(builtin.encode("ascii")) for builtin in builtin_list],
        hinted_class_hash=compute_hinted_class_hash(contract_class=contract_class),
        bytecode_length=len(contract_class.program.data),
        bytecode_ptr=contract_class.program.data,
    )

def run_function_runner(
        runner,
        program,
        func_name: str,
        *args,
        hint_locals: Optional[Dict[str, Any]] = None,
        static_locals: Optional[Dict[str, Any]] = None,
        verify_secure: Optional[bool] = None,
        trace_on_failure: bool = False,
        apply_modulo_to_args: Optional[bool] = None,
        use_full_name: bool = False,
        verify_implicit_args_segment: bool = False,
        **kwargs,
    ) -> Tuple[Tuple[MaybeRelocatable, ...], Tuple[MaybeRelocatable, ...]]:
        """
        Runs func_name(*args).
        args are converted to Cairo-friendly ones using gen_arg.

        Returns the return values of the function, splitted into 2 tuples of implicit values and
        explicit values. Structs will be flattened to a sequence of felts as part of the returned
        tuple.

        Additional params:
        verify_secure - Run verify_secure_runner to do extra verifications.
        trace_on_failure - Run the tracer in case of failure to help debugging.
        apply_modulo_to_args - Apply modulo operation on integer arguments.
        use_full_name - Treat 'func_name' as a fully qualified identifier name, rather than a
          relative one.
        verify_implicit_args_segment - For each implicit argument, verify that the argument and the
          return value are in the same segment.
        """
        assert isinstance(program, Program)
        entrypoint = program.get_label(func_name, full_name_lookup=use_full_name)

        #Construct Fu
        structs_factory = CairoStructFactory.from_program(program=program)
        func = ScopedName.from_string(scope=func_name)

        full_args_struct = structs_factory.build_func_args(func=func)
        all_args = full_args_struct(*args, **kwargs)

        try:
            runner.run_from_entrypoint(
                entrypoint,
                all_args,
                typed_args=True,
                hint_locals=hint_locals,
                static_locals=static_locals,
                verify_secure=verify_secure,
                apply_modulo_to_args=apply_modulo_to_args,
            )
        except (VmException, SecurityError, AssertionError) as ex:
            if trace_on_failure:
                print(
                    f"""\
Got {type(ex).__name__} exception during the execution of {func_name}:
{str(ex)}
"""
                )
                #trace_runner(runner=runner)
            raise

        # The number of implicit arguments is identical to the number of implicit return values.
        n_implicit_ret_vals = structs_factory.get_implicit_args_length(func=func)
        n_explicit_ret_vals = structs_factory.get_explicit_return_values_length(func=func)
        n_ret_vals = n_explicit_ret_vals + n_implicit_ret_vals
        implicit_retvals = tuple(
            runner.get_range(
                runner.get_ap() - n_ret_vals, n_implicit_ret_vals
            )
        )

        explicit_retvals = tuple(
            runner.get_range(
                runner.get_ap() - n_explicit_ret_vals, n_explicit_ret_vals
            )
        )

        # Verify the memory segments of the implicit arguments.
        if verify_implicit_args_segment:
            implicit_args = all_args[:n_implicit_ret_vals]
            for implicit_arg, implicit_retval in safe_zip(implicit_args, implicit_retvals):
                assert isinstance(
                    implicit_arg, RelocatableValue
                ), f"Implicit arguments must be RelocatableValues, {implicit_arg} is not."
                assert isinstance(implicit_retval, RelocatableValue), (
                    f"Argument {implicit_arg} is a RelocatableValue, but the returned value "
                    f"{implicit_retval} is not."
                )
                assert implicit_arg.segment_index == implicit_retval.segment_index, (
                    f"Implicit argument {implicit_arg} is not on the same segment as the returned "
                    f"{implicit_retval}."
                )
                assert implicit_retval.offset >= implicit_arg.offset, (
                    f"The offset of the returned implicit argument {implicit_retval} is less than "
                    f"the offset of the input {implicit_arg}."
                )

        return implicit_retvals, explicit_retvals
