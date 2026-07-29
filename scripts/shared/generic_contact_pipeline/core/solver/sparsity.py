from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ResidualRowDependency:
    """State-frame dependency for a contiguous row range within one factor."""

    factor_id: str
    residual_start: int
    residual_stop: int
    frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.factor_id or self.residual_start < 0 or self.residual_stop <= self.residual_start:
            raise ValueError("residual row dependency requires a factor and positive half-open row range")
        if not self.frames or len(set(self.frames)) != len(self.frames):
            raise ValueError("residual row dependency requires unique state frames")


def build_factor_frame_jacobian_sparsity(
    *,
    factor_block_sizes: Sequence[tuple[str, int]],
    frames: tuple[int, ...],
    parameter_width_per_frame: int,
    dependencies: Sequence[ResidualRowDependency],
):
    """Compile row-level factor dependencies into SciPy Jacobian sparsity.

    Factors without explicit dependencies remain dense. Once a factor has any
    dependency declarations, those declarations must cover its residual block
    exactly once. This prevents an incorrect sparse declaration from silently
    hiding a real state dependency from finite differencing.
    """

    try:
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - depends on declared runtime
        raise RuntimeError("factor-frame sparsity requires scipy") from exc

    if not frames or parameter_width_per_frame <= 0:
        raise ValueError("Jacobian sparsity requires frames and a positive per-frame parameter width")
    frame_columns = {
        frame: range(index * parameter_width_per_frame, (index + 1) * parameter_width_per_frame)
        for index, frame in enumerate(frames)
    }
    if len(frame_columns) != len(frames):
        raise ValueError("Jacobian sparsity frames must be unique")
    known_factors = {factor_id for factor_id, _ in factor_block_sizes}
    if len(known_factors) != len(factor_block_sizes) or any(size <= 0 for _, size in factor_block_sizes):
        raise ValueError("Jacobian sparsity factor blocks must be unique and nonempty")
    unknown_factors = {dependency.factor_id for dependency in dependencies} - known_factors
    unknown_frames = {frame for dependency in dependencies for frame in dependency.frames} - set(frames)
    if unknown_factors:
        raise ValueError("Jacobian dependency references unknown factors: " + ",".join(sorted(unknown_factors)))
    if unknown_frames:
        raise ValueError("Jacobian dependency references unknown frames: " + ",".join(map(str, sorted(unknown_frames))))

    dependencies_by_factor: dict[str, list[ResidualRowDependency]] = {}
    for dependency in dependencies:
        dependencies_by_factor.setdefault(dependency.factor_id, []).append(dependency)

    total_rows = sum(size for _, size in factor_block_sizes)
    total_columns = len(frames) * parameter_width_per_frame
    sparsity = lil_matrix((total_rows, total_columns), dtype=bool)
    row_offset = 0
    for factor_id, block_size in factor_block_sizes:
        factor_dependencies = dependencies_by_factor.get(factor_id, [])
        if not factor_dependencies:
            sparsity[row_offset : row_offset + block_size, :] = True
            row_offset += block_size
            continue
        coverage = [0] * block_size
        for dependency in factor_dependencies:
            if dependency.residual_stop > block_size:
                raise ValueError(f"Jacobian dependency exceeds residual block for {factor_id}")
            for residual_index in range(dependency.residual_start, dependency.residual_stop):
                coverage[residual_index] += 1
            columns = [column for frame in dependency.frames for column in frame_columns[frame]]
            sparsity[
                row_offset + dependency.residual_start : row_offset + dependency.residual_stop,
                columns,
            ] = True
        if any(count != 1 for count in coverage):
            raise ValueError(f"Jacobian dependencies must cover each residual row exactly once for {factor_id}")
        row_offset += block_size
    return sparsity.tocsr()
