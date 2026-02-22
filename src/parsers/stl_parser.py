# ============================================================================
# HybridRAG -- STL Parser (src/parsers/stl_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads STL (3D printing/mesh) files and extracts metadata about the
#   3D model. STL files contain triangle meshes -- they define the shape
#   of a 3D object as thousands of tiny triangles.
#
#   STL files contain NO text (no labels, no annotations). They are pure
#   geometry. So for RAG purposes, we extract structural metadata:
#     - Solid name (if present in ASCII STL header)
#     - Triangle count, vertex count
#     - Bounding box dimensions (X/Y/Z size)
#     - Volume estimate
#
#   This metadata helps answer questions like "How big is the part?" or
#   "How many triangles in the mesh?" and helps catalog 3D models.
#
# DEPENDENCIES:
#   pip install numpy-stl  (BSD-3 license)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class StlParser:
    """
    Extract metadata from STL 3D mesh files.

    NON-PROGRAMMER NOTE:
      STL files have no readable text inside them -- they are lists of
      triangle coordinates. We extract dimensional and structural info
      as text so it can be indexed and searched.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "StlParser"}

        try:
            from stl import mesh as stl_mesh  # numpy-stl
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install numpy-stl"
            )
            return "", details

        try:
            m = stl_mesh.Mesh.from_file(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read STL: {e}"
            return "", details

        try:
            # Solid name (ASCII STL files often have a name on the first line)
            name = getattr(m, "name", b"")
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore").strip()
            name = (name or "").strip()

            tri_count = len(m.vectors)
            vert_count = tri_count * 3

            # Bounding box: min/max of all vertices
            mins = m.vectors.reshape(-1, 3).min(axis=0)
            maxs = m.vectors.reshape(-1, 3).max(axis=0)
            dims = maxs - mins

            # Volume estimate (signed volume from triangle normals)
            # This works for watertight meshes
            volume = float(m.get_mass_properties()[0]) if hasattr(m, "get_mass_properties") else 0.0

            parts = [
                f"3D Model (STL): {path.name}",
            ]
            if name:
                parts.append(f"Solid name: {name}")
            parts.extend([
                f"Triangles: {tri_count:,}",
                f"Vertices: {vert_count:,}",
                f"Bounding box: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}",
                f"X range: {mins[0]:.2f} to {maxs[0]:.2f}",
                f"Y range: {mins[1]:.2f} to {maxs[1]:.2f}",
                f"Z range: {mins[2]:.2f} to {maxs[2]:.2f}",
            ])
            if volume > 0:
                parts.append(f"Volume: {volume:.2f}")

            full = "\n".join(parts)
            details["total_len"] = len(full)
            details["triangles"] = tri_count
            return full, details

        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"
            return "", details
