import json


class ParseError(Exception):
    pass


def parse_any(x):
    return x

def parse_bool(x):
    if type(x) != bool:
        raise ParseError(f"expected an bool but got {repr(x)}")
    return x

def parse_int(x):
    if type(x) != int:
        raise ParseError(f"expected an int but got {repr(x)}")
    return x

def parse_float(x):
    if type(x) != float:
        raise ParseError(f"expected an float but got {repr(x)}")
    return x

def parse_str(x):
    if type(x) != str:
        raise ParseError(f"expected a str but got {repr(x)}")
    return x

#a list of items each with the same validator
def parse_listof(item_validator):
    def parse_listof_impl(x):
        if type(x) != list:
            raise ParseError(f"expected a list but got {repr(x)}")
        return [item_validator(y) for y in x]
    return parse_listof_impl


##{
##    "a" : (parse, "required"),
##    "b" : (parse, "implicit", 69),
##    "c" : (parse, "optional")
##}


class parse_dict():
    def __init__(self, items):
        assert type(items) == dict
        for key, item_t in items.items():
            assert type(key) == str
            assert type(item_t) == tuple
            assert len(item_t) >= 2
            assert callable(item_t[0]) #the parser for the item
            assert item_t[1] in {"required", "implicit", "optional"}
            if item_t[1] == "implicit":
                assert len(item_t) == 3
                item_t[0](item_t[2]) #validate the implicit value
            else:
                assert len(item_t) == 2
        self._items = items

    def get_items(self):
        return self._items

    def __call__(self, x):
        if type(x) != dict:
            raise ParseError(f"expected a dict but got {repr(x)}")

        for k in x:
            if not k in self._items:
                raise ParseError(f"keys of {repr(x)} should be a subset of {repr(set(self._items.keys()))}")

        y = {}
        for key, item_t in self._items.items():
            parse = item_t[0]
            mode = item_t[1]
            if mode == "required":
                if not key in x:
                    raise ParseError(f"required key {repr(key)} is not present in {repr(x)}")
                y[key] = parse(x[key])
            elif mode == "implicit":
                if key in x:
                    y[key] = parse(x[key])
                else:
                    y[key] = parse(item_t[2])
            elif mode == "optional":
                if key in x:
                    y[key] = parse(x[key])
            else:
                assert False
            
        return y


#a dict where the keys are arbitrary and all the values have a common validator
def parse_vardict(item_validator):
    assert callable(item_validator)
    def parse_vardict_impl(x):
        if type(x) != dict:
            raise ParseError(f"expected a dict but got {repr(x)}")
        for key in x:
            if type(key) != str:
                raise ParseError(f"expected a str for key but got {repr(x)}")
        return {k : item_validator(v) for k, v in x.items()}
    return parse_vardict_impl

#a str which can take on one of a finite set of values
def parse_enum(allowed_values):
    assert type(allowed_values) == set
    for allowed_value in allowed_values:
        assert type(allowed_value) == str
    def parse_enum_impl(x):
        if type(x) != str:
            raise ParseError(f"expected a str for enum value but got {repr(x)}")
        if not x in allowed_values:
            raise ParseError(f"expected {repr(x)} to be one of {repr(allowed_values)}")
        return x
    return parse_enum_impl



#a dict whose keys depend on the value of some special key
def parse_variant(base_dict, varient_key, variant_dicts):
    assert type(base_dict) == parse_dict
    base_keys = set()
    base_keys |= base_dict.get_items().keys()
    assert type(varient_key) == str
    assert not varient_key in base_keys
    for key, var_dict in variant_dicts.items():
        assert type(key) == str
        if key in base_keys or key == varient_key:
            assert False
        assert type(var_dict) == parse_dict

    def parse_variant_impl(x):
        if type(x) != dict:
            raise ParseError(f"expected a dict but got {repr(x)}")
        if not varient_key in x:
            raise ParseError(f"{repr(x)} does not contain varient key {repr(varient_key)}")
        v = x[varient_key]
        if not v in variant_dicts:
            raise ParseError(f"varient {repr(v)} is not a valid value in {repr(set(variant_dicts.keys()))}")
        var_dict = variant_dicts[v]
        var_keys = var_dict.get_items().keys()
         
        x_base = base_dict({k : val for k, val in x.items() if k in base_keys})
        x_rest = var_dict({k : val for k, val in x.items() if k in var_keys})
        for k in x:
            if not k in x_base and not k in x_rest and not k == varient_key:
                raise ParseError(f"unknown key {repr(k)}")
        
        return x_base | {varient_key : v} | x_rest
        
    return parse_variant_impl

        
    

class Context():
    def __init__(self, data):
        parse_content_type = parse_variant(
            parse_dict({"default" : (parse_any, "optional"),
                        "optional" : (parse_bool, "implicit", False)}),
            "kind",
            {
                "bool" : parse_dict({}),
                "int" : parse_dict({}),
                "float" : parse_dict({}),
                "str" : parse_dict({}),
                "list" : parse_dict({"item_t" : (lambda x : parse_content_type(x), "required")}),
                "inst" : parse_dict({"type" : (parse_str, "required")}), #unique pointer to an object of this ident
                "ptr" : parse_dict({"type" : (parse_str, "required"), "target" : (parse_str, "required")}) #shared pointer to the specified target of an object of this ident
            })
        
        parse_ctx = parse_listof(parse_dict(
            {
                "type" : (parse_str, "required"),
                "super" : (parse_listof(parse_str), "implicit", []),
                "abstract" : (parse_bool, "implicit", False),
                "targets" : (parse_listof(parse_str), "implicit", []), #the places which pointers can point to
                "content" : (parse_vardict(parse_content_type), "implicit", {})
                
            }))

        ctx = parse_ctx(data)

        #check that 'type' values are unique
        types = [t["type"] for t in ctx]
        for tn in types:
            if types.count(tn) != 1:
                raise ParseError(f"type names should be unique but {repr(tn)} appears {types.count(tn)} times")

        #create lookup types by their 'type' field
        ctx = {t["type"] : t for t in ctx}

        #compute and validate super types
        supers = {}
        for tn in types:
            supers[tn] = set([tn])
            for sn in ctx[tn]["super"]:
                if not sn in supers:
                    raise ParseError(f"type {repr(tn)} inherits {repr(sn)} which is not defined")
                supers[tn] |= supers[sn]

        #check that supers dont cause content name clashes or target name clashes
        for sns in supers.values():
            used_content_keys = set()
            for sn in sns:
                for k in ctx[sn]["content"].keys():
                    if k in used_content_keys:
                        raise ParseError(f"content key {repr(k)} is used by multiple types in an inheritance chain")
                    used_content_keys.add(k)

            used_target_keys = set()
            for sn in sns:
                for k in ctx[sn]["targets"]:
                    if k in used_target_keys:
                        raise ParseError(f"target {repr(k)} is used by multiple types in an inheritance chain")
                    used_target_keys.add(k)

        #check that ptrs point to valid targets
        def validate_insts_and_ptrs(typ):
            kind = typ["kind"]
            if kind == "list":
                validate_insts_and_ptrs(typ["item_t"])
            elif kind in {"bool", "int", "float", "str"}:
                pass #nothing to check here
            elif kind == "ptr":
                if not typ["type"] in ctx:
                    raise ParseError(f"{repr(typ['type'])} is not valid as a pointer type")
                if not typ["target"] in ctx[typ["type"]]["targets"]:
                    raise ParseError(f"{repr(typ['target'])} is not a valid target for {repr(typ['type'])}")
            elif kind == "inst":
                if not typ["type"] in ctx:
                    raise ParseError(f"{repr(typ['type'])} is not valid as a pointer type")
            else:
                raise Exception(f"unexpected kind {repr(kind)}")
            
        for tn in types:
            for key, typ in ctx[tn]["content"].items():
                validate_insts_and_ptrs(typ)

        self._ctx = ctx
        self._supers = supers

    def get_all_types(self):
        return set(self._ctx.keys())
    
    def get_type(self, tn):
        return self._ctx[tn]

    def get_supers(self, tn):
        return self._supers[tn]

    def get_super_content(self, tn):
        content = {}
        for sn in self.get_supers(tn):
            for key, val_t in self.get_type(sn)["content"].items():
                assert not key in content
                content[key] = val_t
        return content

    def get_target_keys(self, tn):
        targets = set()
        for sn in self.get_supers(tn):
            for targ in self.get_type(sn)["targets"]:
                assert not targ in targets
                targets.add(targ)
        return targets

    def is_type(self, tn):
        return tn in self._ctx




def parse_root(root_t, data):
    #accumulate a set of (obj, tn, targ, ident)
    #once all objects are parsed, we compare this set against target_index to check that all points are pointing to valid things
    existing_pointers = set()

    target_owners = {tn : {targ : {} for targ in ctx.get_type(tn)["targets"]} for tn in ctx.get_all_types()}
        
    #create a perser for an object of type obj_t
    class ObjectBase():
        pass
    
    def parse_object(obj_t, owner): 
        #the parser
        class Object(ObjectBase):
            def __init__(self, obj):
                assert owner is None or isinstance(owner, ObjectBase)
                
                if not "type" in obj:
                    raise ParseError(f"object {repr(obj)} is missing a 'type' field")
                tn = obj["type"]
                if not ctx.is_type(tn):
                    raise ParseError(f"{repr(tn)} is not a valid type: {repr(ctx.get_all_types())}")
                if ctx.get_type(tn)["abstract"]:
                    raise Exception(f"cannot make abstract type {repr(obj_t)}")
                
                if not obj_t in ctx.get_supers(tn):
                    raise ParseError(f"invalid object type: {repr(tn)} does not inherit from {repr(obj_t)}")
                
                def parser_from_ctx_t(ctx_t):
                    kind = ctx_t["kind"]
                    if kind == "list":
                        return parse_listof(parser_from_ctx_t(ctx_t["item_t"]))
                    elif kind == "inst":
                        return parse_object(ctx_t["type"], self)
                    elif kind == "ptr":
                        def parse_ptr(p):
                            p = parse_str(p)
                            existing_pointers.add((self, ctx_t["type"], ctx_t["target"], p))
                            return lambda : target_owners[ctx_t["type"]][ctx_t["target"]][p]
                        return parse_ptr
                    elif kind == "bool":
                        return parse_bool
                    elif kind == "int":
                        return parse_int
                    elif kind == "float":
                        return parse_float
                    elif kind == "str":
                        return parse_str
                    else:
                        raise NotImplementedError(f"unknown kind {repr(kind)}")
                
                opt = {}
                targs = ctx.get_target_keys(tn)

                content_items = {}
                for key, val_t in ctx.get_super_content(tn).items():
                    if val_t["optional"] and "default" in val_t:
                        raise ParseError("Cannot have a field which is both optional and has a default value")
                    if val_t["optional"]:
                        content_items[key] = (parser_from_ctx_t(val_t), "optional")
                    elif "default" in val_t:
                        content_items[key] = (parser_from_ctx_t(val_t), "implicit", val_t["default"])                        
                    else:
                        content_items[key] = (parser_from_ctx_t(val_t), "required")
                items = {"type" : (parse_str, "required"),
                         "content" : (parse_dict(content_items), "required")}
                if len(targs) == 0:
                    items["targets"] = (parse_dict({targ : (parse_str, "required") for targ in targs}), "implicit", {})
                else:
                    items["targets"] = (parse_dict({targ : (parse_str, "required") for targ in targs}), "required")
                assert items.keys() == {"type", "content", "targets"}
                obj = parse_dict(items)(obj)
                
                #check that the target idents are distinct
                for sn in ctx.get_supers(tn):
                    for targ in ctx.get_type(sn)["targets"]:
                        ident = obj["targets"][targ]
                        if ident in target_owners[sn][targ]:
                            raise ParseError(f"target of type {repr(sn)}->{repr(targ)} has ident {repr(ident)} which is already in use")
                        target_owners[sn][targ][ident] = self

                self._tn = tn
                self._content = obj["content"]
                self._targets = obj["targets"]
                self._owner = owner
                self._reverse_pointers = {target : [] for target in self._targets}

            def __repr__(self):
                return self._tn + "(" + ",".join(k + "=" + str(v) for k, v in self._content.items()) + ")"

            def __getitem__(self, key):
                return self._content[key]

            def to_json(self):
                content_t = ctx.get_super_content(self._tn)
                
                def field_to_json(value, value_t):
                    kind = value_t["kind"]
                    if kind == "list":
                        return [field_to_json(x, value_t["item_t"]) for x in value]
                    elif kind == "inst":
                        return value.to_json()
                    elif kind == "ptr":
                        #value() is the object we are pointing at
                        #value().get_target(value_t["target"]) is the id of the target we are pointing at
                        return value().get_target(value_t["target"])
                    elif kind in {"bool", "int", "float", "str", "ptr"}:
                        return value
                    else:
                        raise NotImplementedError(kind)

                content = {}
                for key in content_t:
                    if key in self._content:
                        content[key] = field_to_json(self._content[key], content_t[key])
                    else:
                        assert content_t[key]["optional"]
                    
                return {"type" : self._tn,
                        "content" : content,
                        "targets" : self._targets}

            def get_owner(self):
                return self._owner

            def get_target(self, targ):
                return self._targets[targ]

            def _add_reverse_pointer(self, targ, obj):
                self._reverse_pointers[targ].append(obj)

            def get_reverse_pointers(self, targ):
                return self._reverse_pointers[targ]
                                    
        return Object

    obj = parse_object(root_t, None)(data)

    #check that pointers all point to things which exist
    for pointing_obj, typ, targ, ident in existing_pointers:
        if not ident in target_owners[typ][targ]:
            raise ParseError(f"pointer {repr(ident)} does not point to an existing target of type {repr(typ)}->{repr(targ)}")
        targ_obj = target_owners[typ][targ][ident]
        targ_obj._add_reverse_pointer(targ, pointing_obj)
            
    return obj, target_owners



class Data():
    def __init__(self, ctx, root_t, root_data):
        assert type(ctx) == Context
        self._ctx = ctx
        self._root_t = root_t
        
        root, target_owners = parse_root(self._root_t, root_data)
        self._root = root
        self._target_owners = target_owners

        self.validate()

    def to_json(self):
        return self._root.to_json()

    def get_root(self):
        return self._root

    def validate(self):
        #this checks that self._root has a valid format, assuming that self._root.to_json() works correctly, which it should
        parse_root(self._root_t, self._root.to_json())
        




if __name__ == "__main__":
    with open("context.txt", "r") as f:
        ctx = Context(json.loads(f.read()))



##    with open("ft3.json", "r") as f:
##        tree = json.loads(f.read())
##
##    def parse(ident):
##        obj = tree[ident]
##
##        t = obj["type"]
##        ans = {"type" : t}
##        
##        content = {}
##        if t == "tree":
##            old_ents = [parse(ob_ident) for ob_ident in obj["content"]["entities"]]
##            content["entities"] = []
##            content["images"] = []
##            for e in old_ents:
##                if e["type"] == "image":
##                    content["images"].append(e)
##                else:
##                    content["entities"].append(e)
##        elif t == "person":
##            content["parent_ordering"] = obj["content"]["parent_ordering"]
##            content["child_ordering"] = obj["content"]["child_ordering"]
##            content["infos"] = [parse(info_ident) for info_ident in obj["content"]["infos"]]
##            ans["targets"] = {"image" : ident, "child_part" : ident, "parent_part" : ident}
##        elif t == "partnership":
##            content["parent_ordering"] = obj["content"]["parent_ordering"]
##            content["child_ordering"] = obj["content"]["child_ordering"]
##            content["infos"] = [parse(info_ident) for info_ident in obj["content"]["infos"]]
##            content["parents"] = [parse(i) for i in obj["content"]["parents"]]
##            content["children"] = [parse(i) for i in obj["content"]["children"]]
##            ans["targets"] = {"image" : ident}
##        elif t == "parent_ptr":
##            content["person"] = obj["content"]["target"]
##        elif t == "child_ptr":
##            content["person"] = obj["content"]["target"]
##            content["adopted"] = obj["content"]["adopted"]
##        elif t == "image":
##            content["infos"] = [parse(info_ident) for info_ident in obj["content"]["infos"]]
##            content["path"] = obj["content"]["path"]
##            content["subimages"] = [parse(subimg_ident) for subimg_ident in obj["content"]["subimages"]]
##        elif t == "subimage":
##            content["entity"] = obj["content"]["entity"]
##            content["x"] = obj["content"]["x"]
##            content["y"] = obj["content"]["y"]
##            content["w"] = obj["content"]["w"]
##            content["h"] = obj["content"]["h"]
##            content["usable"] = obj["content"]["usable"]
##        elif t == "subinfo":
##            content["title"] = obj["content"]["title"]
##            content["infos"] = [parse(info_ident) for info_ident in obj["content"]["infos"]]
##        elif t == "string":
##            content["string"] = obj["content"]["string"]
##        elif t == "date":
##            if "day" in obj["content"]:
##                content["day"] = obj["content"]["day"]
##            if "month" in obj["content"]:
##                content["month"] = obj["content"]["month"]
##            if "year" in obj["content"]:
##                content["year"] = obj["content"]["year"]
##            content["tags"] = obj["content"]["tags"]
##        else:
##            raise NotImplementedError(t)
##
##        ans["content"] = content
##        
##        return ans
##
##    d = parse("0")
##
##    with open("sus.json", "w") as g:
##        g.write(json.dumps(d, indent = 2))
##
##    print("oop")
##    
##    data = Data(ctx, "tree", d)
##    print("YAY", type(data))

        
    with open("ftng.json", "r") as f:
        data = Data(ctx, "tree", json.loads(f.read()))

    print(data)
    print(data.get_root()["entities"][100].get_reverse_pointers("child_part")[0]["person"]().get_target("child_part"))






















