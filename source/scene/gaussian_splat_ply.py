"""Gaussian splat PLY 的轻量检查工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TYPE_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
_FLOAT32_TYPES = {"float", "float32"}
_GAUSSIAN_REQUIRED_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


@dataclass(frozen=True)
class PlyProperty:
    """记录 PLY 中单个 vertex property。"""

    name: str
    type_name: str
    is_list: bool = False


@dataclass(frozen=True)
class PlyHeader:
    """只保存本项目需要的 PLY header 信息。"""

    path: Path
    format_name: str
    vertex_count: int
    vertex_properties: tuple[PlyProperty, ...]
    has_faces: bool
    data_offset: int
    comments: tuple[str, ...]

    @property
    def vertex_property_names(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.vertex_properties)

    @property
    def vertex_stride_bytes(self) -> int:
        total = 0
        for prop in self.vertex_properties:
            if prop.is_list:
                raise ValueError("vertex list property 不支持固定 stride 解析")
            total += _TYPE_SIZES[prop.type_name]
        return total

    @property
    def all_vertex_properties_float32(self) -> bool:
        return all((not prop.is_list) and prop.type_name in _FLOAT32_TYPES for prop in self.vertex_properties)


def _normalize_type(raw_type: str) -> str:
    type_name = raw_type.strip().lower()
    if type_name not in _TYPE_SIZES:
        raise ValueError(f"不支持的 PLY property 类型: {raw_type}")
    return type_name


def parse_ply_header(path: str | Path) -> PlyHeader:
    """解析 PLY header，不读取后续大体积点数据。"""

    ply_path = Path(path).expanduser().resolve()
    with ply_path.open("rb") as stream:
        first_line = stream.readline().decode("ascii", errors="replace").strip()
        if first_line != "ply":
            raise ValueError(f"不是 PLY 文件: {ply_path}")

        format_name = ""
        vertex_count = 0
        has_faces = False
        vertex_properties: list[PlyProperty] = []
        comments: list[str] = []
        current_element: str | None = None

        while True:
            raw_line = stream.readline()
            if raw_line == b"":
                raise ValueError(f"PLY header 缺少 end_header: {ply_path}")
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                data_offset = stream.tell()
                break
            if not line:
                continue

            parts = line.split()
            keyword = parts[0]
            if keyword == "comment":
                comments.append(line.removeprefix("comment").strip())
            elif keyword == "format" and len(parts) >= 2:
                format_name = parts[1]
            elif keyword == "element" and len(parts) >= 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    has_faces = True
            elif keyword == "property" and current_element == "vertex":
                if len(parts) >= 5 and parts[1] == "list":
                    vertex_properties.append(
                        PlyProperty(
                            name=parts[4],
                            type_name=_normalize_type(parts[3]),
                            is_list=True,
                        )
                    )
                elif len(parts) >= 3:
                    vertex_properties.append(
                        PlyProperty(
                            name=parts[2],
                            type_name=_normalize_type(parts[1]),
                        )
                    )

    if not format_name:
        raise ValueError(f"PLY header 缺少 format: {ply_path}")
    if vertex_count <= 0:
        raise ValueError(f"PLY header 缺少有效 vertex 数量: {ply_path}")
    return PlyHeader(
        path=ply_path,
        format_name=format_name,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        has_faces=has_faces,
        data_offset=data_offset,
        comments=tuple(comments),
    )


def is_gaussian_splat_ply(header: PlyHeader) -> bool:
    """判断 PLY 是否包含 3DGS 常见属性。"""

    return _GAUSSIAN_REQUIRED_PROPERTIES.issubset(set(header.vertex_property_names))


def describe_ply(path: str | Path) -> dict[str, Any]:
    """返回适合写入报告的 PLY 摘要。"""

    header = parse_ply_header(path)
    return {
        "path": str(header.path),
        "format": header.format_name,
        "vertex_count": header.vertex_count,
        "vertex_property_count": len(header.vertex_properties),
        "vertex_properties": list(header.vertex_property_names),
        "has_faces": header.has_faces,
        "data_offset": header.data_offset,
        "vertex_stride_bytes": header.vertex_stride_bytes,
        "all_vertex_properties_float32": header.all_vertex_properties_float32,
        "is_gaussian_splat_ply": is_gaussian_splat_ply(header),
    }
