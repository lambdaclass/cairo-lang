from cairo_rs_py import CairoRunner
from starkware.cairo.common.cairo_function_runner import CairoFunctionRunner, run_function_runner
from starkware.starknet.core.os.contract_class.class_hash_utils import (
    get_contract_class_struct,
    load_contract_class_cairo_program,
)
from starkware.starknet.core.os.contract_class.utils import ClassHashType, class_hash_cache_ctx_var
from starkware.starknet.public.abi import starknet_keccak
from starkware.starknet.services.api.contract_class.contract_class import ContractClass


def compute_class_hash(contract_class: ContractClass) -> int:
    cache = class_hash_cache_ctx_var.get()
    if cache is None:
        return _compute_class_hash_inner(contract_class=contract_class)

    contract_class_bytes = contract_class.dumps(sort_keys=True).encode()
    key = (ClassHashType.CONTRACT_CLASS, starknet_keccak(data=contract_class_bytes))

    if key not in cache:
        cache[key] = _compute_class_hash_inner(contract_class=contract_class)

    return cache[key]


def _compute_class_hash_inner(contract_class: ContractClass) -> int:
    program = load_contract_class_cairo_program()
    contract_class_struct = get_contract_class_struct(
        identifiers=program.identifiers, contract_class=contract_class
    )
    runner = CairoRunner(program=program.dumps(), entrypoint=None)
    runner.initialize_function_runner(add_segment_arena_builtin=False)
    poseidon_ptr = runner.get_poseidon_builtin_base()
    range_check_ptr = runner.get_range_check_builtin_base()

    run_function_runner(
        "starkware.starknet.core.os.contract_class.contract_class.class_hash",
        poseidon_ptr=poseidon_ptr,
        range_check_ptr=range_check_ptr,
        contract_class=contract_class_struct,
        use_full_name=True,
        verify_secure=False,
    )
    _, class_hash = runner.get_return_values(2)
    return class_hash
