import os.path
import glob
import types
import itertools
import subprocess
import sys
import json
import ctypes
import numpy as np


def init(run_cargo, cache_path):
    def generate_type_combinations(export):
        keys = tuple(export.ty.keys())
        for ty in itertools.product(*[export.ty[k] for k in keys]):
            yield {keys[i][0]:ty[i] for i in range(len(keys))}

    def get_type_string(ty, order):
        return "".join([f"_{ty[order[i]]}" for i in range(len(order))])
    
    lib_path = "rust/target/release/libaethon.so"
    if os.path.exists(lib_path):
        lib_modified_time = os.path.getmtime(lib_path)
        for fn in glob.iglob("rust/src/*.rs"):
            if os.path.getmtime(fn) > lib_modified_time:
                run_cargo = True
    else:
        run_cargo = True
    
    if run_cargo:
        exports = []
        for src_fn in glob.iglob("rust/src/*.rs"):
            fn = src_fn.split("/")[-1].split(".")[0]
            if fn == "exports":
                continue
            with open(src_fn, "r") as f:
                content = "".join(f.readlines()).replace("\n", " ")
            start = content.find("//PYTHON_EXPORT")
            while start != -1:
                end = content.find("{", start)
                exports.append((fn, content[start:end]))
                start = content.find("//PYTHON_EXPORT", end)
        
        for i, (fn, content) in enumerate(exports):
            export = types.SimpleNamespace(fn=fn, contains_ndarrays=False)
            
            index = content.find("pub fn ")
            assert index != - 1
            ty, content = content[15:index].strip(), content[index+7:]
            export.ty = []
            if len(ty) > 0:
                for ty in ty.split(";"):
                    export.ty.append([
                        ty.strip() for ty in ty.split(",")
                    ])
            
            index = content.find("(")
            assert index != -1
            export.name, content = content[:index], content[index+1:]
            index = export.name.find("<")
            if index != -1:
                export.name, generics = export.name[:index], export.name[index+1:]
                for j, ty in enumerate(generics.split(",")):
                    export.ty[j].append(ty.split(":")[0].strip())
                export.ty_order = {j:ty[-1] for j,ty in enumerate(export.ty)}
                export.ty = {ty[-1]:[*ty[:-1]] for ty in export.ty}
            
            index = content.find(")")
            assert index != -1
            params, content = content[:index], content[index+1:]
            export.params = []
            for param in params.split(","):
                index = param.find(":")
                if index == -1:
                    continue
                param = param[index+1:].strip()
                while "  " in param:
                    param = param.replace("  ", " ")
                index = param.find("<")
                if index != -1:
                    export.contains_ndarrays = True
                    param = param[:index].strip(), param[index+1:-1].strip()
                    assert param[0] == "NdArray"
                else:
                    param = param,
                export.params.append(param)
            
            index = content.find("->")
            export.ret_type = content[index+2:].strip() if index != -1 else None
            
            exports[i] = export
            
        def generate_export(f, export, ty):
            type_string = get_type_string(ty, getattr(export, "ty_order", {}))
            f.write(f"#[no_mangle]\npub extern \"C\" fn _{export.name}{type_string}(\n")
            for i, param in enumerate(export.params):
                suffix = "," if i < len(export.params)-1 else ""
                if len(param) > 1:
                    t = ty[param[1]] if param[1] in ty else param[1]
                    f.write(f"\tp{i}: *const {t}, p{i}_shape: *const i32, p{i}_strides: *const i32, p{i}_ndims: i32{suffix}\n")
                else:
                    f.write(f"\tp{i}: {param[0]}{suffix}\n")
            f.write(") ")
            if not export.ret_type is None:
                f.write(f"-> {export.ret_type} ")
            f.write("{\n")
            if export.contains_ndarrays:
                for i, param in enumerate(export.params):
                    if len(param) == 1:
                        continue
                    f.write(f"\tlet p{i} = NdArray::from_ptr(p{i}, p{i}_shape, p{i}_strides, p{i}_ndims);\n")
                f.write("\n")
            f.write(f"\t{export.name}(")
            params = ", ".join([f"p{i}" for i in range(len(export.params))])
            f.write(params)
            f.write(")\n}\n\n")
            
        with open("rust/src/exports.rs", "w") as f:
            for fn in set([export.fn for export in exports]):
                funcs = "{%s}"%",".join([export.name for export in exports if export.fn == fn])
                f.write(f"use crate::{fn}::{funcs};\n")
            f.write("use crate::utils::*;\n\n")
            for export in exports:
                if len(export.ty) > 0:
                    for ty in generate_type_combinations(export):
                        generate_export(f, export, ty)
                else:
                    generate_export(f, export, {})
        
        ret_val = subprocess.call("cargo build --release --manifest-path rust/Cargo.toml", shell=True)
        if ret_val != 0:
            sys.exit(ret_val)

        with open(f"{cache_path}/rust_exports.json", "w") as f:
            json.dump([export.__dict__ for export in exports], f)
    else:
        with open(f"{cache_path}/rust_exports.json", "r") as f:
            exports = [types.SimpleNamespace(**export) for export in json.load(f)]
        for export in exports:
            if not hasattr(export, "ty_order"):
                continue
            export.ty_order = {int(k):v for k,v in export.ty_order.items()}
            
    type_map = {
        "bool": (ctypes.c_bool, None),
        "i8": (ctypes.c_int8, np.int8),
        "i16": (ctypes.c_int16, np.int16),
        "i32": (ctypes.c_int32, np.int32),
        "i64": (ctypes.c_int64, np.int64),
        "isize": (ctypes.c_ssize_t, None),
        "u8": (ctypes.c_uint8, np.uint8),
        "u16": (ctypes.c_uint16, np.uint16),
        "u32": (ctypes.c_uint32, np.uint32),
        "u64": (ctypes.c_uint64, np.uint64),
        "usize": (ctypes.c_size_t, None),
        "f32": (ctypes.c_float, np.float32),
        "f64": (ctypes.c_double, np.float64)
    }
    reverse_type_map = {np.dtype(numpy_type):rust_type for rust_type, (_, numpy_type) in type_map.items() if not numpy_type is None}
    type_map = {rust_type:python_type for rust_type, (python_type, _) in type_map.items()}

    lib = ctypes.CDLL(lib_path)
    
    class FunctionWrapper():
        def __init__(self, export, name_suffix="", params=None):
            self.func = lib[f"_{export.name}{name_suffix}"]
            params = export.params if params is None else params
            argtypes = []
            for param in params:
                if len(param) > 1:
                    argtypes.extend([
                        np.ctypeslib.ndpointer(type_map[param[1]]),
                        np.ctypeslib.ndpointer(ctypes.c_int32),
                        np.ctypeslib.ndpointer(ctypes.c_int32),
                        ctypes.c_int32
                    ])
                else:
                    argtypes.append(type_map[param[0]])
            self.func.argtypes = argtypes
            if not export.ret_type is None:
                self.func.restype = type_map[export.ret_type]
        
        def __call__(self, *args):
            rust_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    rust_args.extend([arg, np.asarray(arg.shape,dtype=np.int32), np.asarray(arg.strides,dtype=np.int32), len(arg.shape)])
                else:
                    rust_args.append(arg)
            return self.func(*rust_args)
    
    class GenericFunctionWrapper():
        def __init__(self, export):
            self.funcs = {}
            for ty in generate_type_combinations(export):
                type_string = get_type_string(ty, export.ty_order)
                params = [(param[0],ty[param[1]]) if len(param)>1 and param[1] in ty else param for param in export.params]
                self.funcs[type_string] = FunctionWrapper(export, type_string, params)
            self.type_args = []
            for i in range(len(export.ty_order)):
                ty = export.ty_order[i]
                j = [j for j,param in enumerate(export.params) if len(param)>1 and param[1]==ty][0]
                self.type_args.append(j)
        
        def __call__(self, *args):
            type_string = ""
            for i in self.type_args:
                t = reverse_type_map[args[i].dtype]
                type_string = f"{type_string}_{t}"
            return self.funcs[type_string](*args)
    
    for export in exports:
        if export.contains_ndarrays:
            if len(export.ty) == 0:
                func = FunctionWrapper(export)
            else:
                func = GenericFunctionWrapper(export)
        else:
            func = lib[f"_{export.name}"]
            func.argtypes = [type_map[param[0]] for param in export.params]
            if not export.ret_type is None:
                func.restype = type_map[export.ret_type]
        globals()[export.name] = func
