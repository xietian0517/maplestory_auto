"""Static PE address mapping and IL2CPP metadata registration validation."""
import struct


class PE:
    def __init__(self, data):
        self.data = data
        pe = struct.unpack_from('<I', data, 0x3c)[0]
        if data[:2] != b'MZ' or data[pe:pe+4] != b'PE\0\0':
            raise ValueError('not a PE')
        machine, count = struct.unpack_from('<HH', data, pe+4)
        optional_size = struct.unpack_from('<H', data, pe+20)[0]
        optional = pe+24
        if machine != 0x8664 or struct.unpack_from('<H', data, optional)[0] != 0x20b:
            raise ValueError('only x64 PE supported')
        self.base = struct.unpack_from('<Q', data, optional+24)[0]
        self.sections = []
        for i in range(count):
            off = optional+optional_size+i*40
            name = data[off:off+8].rstrip(b'\0').decode('ascii')
            virtual_size, rva, size, raw = struct.unpack_from('<IIII', data, off+8)
            characteristics = struct.unpack_from('<I', data, off+36)[0]
            self.sections.append(dict(name=name, rva=rva, size=size, raw=raw, virtual_size=virtual_size,
                                      executable=bool(characteristics & 0x20000000)))

    def va(self, raw):
        for section in self.sections:
            if section['raw'] <= raw < section['raw']+section['size']:
                return self.base+section['rva']+raw-section['raw']
        raise ValueError('file offset outside sections')

    def codegen_module(self, name, max_token):
        candidates, start = [], 0
        while True:
            off = self.data.find(name.encode()+b'\0',start)
            if off < 0:
                break
            start = off+1
            try:
                marker = struct.pack('<Q',self.va(off))
            except ValueError:
                continue
            pos = 0
            while True:
                pos = self.data.find(marker,pos)
                if pos < 0:
                    break
                if pos+24 <= len(self.data):
                    _,count,table = struct.unpack_from('<QQQ',self.data,pos)
                    if max_token <= count <= max_token+16:
                        try:
                            raw = self.raw(table,count*8)
                            pointers = struct.unpack_from(f'<{count}Q',self.data,raw)
                            valid = all(p == 0 or any(s['executable'] and self.base+s['rva'] <= p < self.base+s['rva']+s['virtual_size'] for s in self.sections) for p in pointers)
                            if valid:
                                candidates.append(dict(file_offset=pos, count=count, pointers=pointers))
                        except ValueError:
                            pass
                pos += 1
        if len(candidates) != 1:
            raise ValueError(f'{name}: expected one codegen module, found {len(candidates)}')
        return candidates[0]

    def raw(self, va, size=1):
        rva = va-self.base
        for section in self.sections:
            delta = rva-section['rva']
            if 0 <= delta and delta+size <= section['size']:
                raw = section['raw']+delta
                if raw+size <= len(self.data):
                    return raw
        raise ValueError(f'VA not backed by PE file: {va:#x}')

    def registration_candidates(self, count):
        marker = struct.pack('<Q', count)
        result, start = [], 0
        while True:
            off = self.data.find(marker, start)
            if off < 0:
                break
            start = off+1
            # fieldOffsetsCount and typeDefinitionsSizesCount must both match.
            if off < 80 or off+48 > len(self.data) or self.data[off+16:off+24] != marker:
                continue
            reg = struct.unpack_from('<16Q', self.data, off-80)
            try:
                if not 0 < reg[6] < 1000000:
                    continue
                self.raw(reg[7], reg[6]*8)
                self.raw(reg[11], count*8)
                self.raw(reg[13], count*8)
                result.append(dict(file_offset=off-80, type_count=reg[6], type_table_va=reg[7],
                                   field_offsets_va=reg[11], type_sizes_va=reg[13]))
            except ValueError:
                continue
        return result


class NativeTypes:
    PRIMITIVES = {1:'System.Void',2:'System.Boolean',3:'System.Char',4:'System.SByte',5:'System.Byte',
                  6:'System.Int16',7:'System.UInt16',8:'System.Int32',9:'System.UInt32',10:'System.Int64',
                  11:'System.UInt64',12:'System.Single',13:'System.Double',14:'System.String',
                  22:'System.TypedReference',24:'System.IntPtr',25:'System.UIntPtr',28:'System.Object'}

    def __init__(self, pe, metadata):
        self.pe, self.metadata = pe, metadata
        candidates = pe.registration_candidates(len(metadata.types))
        if len(candidates) != 1:
            raise ValueError(f'expected one registration candidate, found {len(candidates)}')
        self.registration = candidates[0]
        self.pointers = self.unpack(self.registration['type_table_va'], f"{self.registration['type_count']}Q")
        self.cache = {}

    def unpack(self, va, fmt):
        return struct.unpack_from('<'+fmt, self.pe.data, self.pe.raw(va, struct.calcsize('<'+fmt)))

    def at(self, va, depth=0):
        if va in self.cache:
            return self.cache[va]
        if depth > 12:
            raise ValueError('native type recursion too deep')
        data, bits = self.unpack(va, 'QI')
        kind = bits >> 16 & 255
        base = None
        if kind in self.PRIMITIVES:
            name = self.PRIMITIVES[kind]
        elif kind in (17,18):
            if data >= len(self.metadata.types):
                raise ValueError('invalid native type definition index')
            base, name = data, self.metadata.types[data]['name']
        elif kind in (15,29):
            child = self.at(data, depth+1)
            name = child['name']+('*' if kind == 15 else '[]')
        elif kind == 21:
            type_ptr, class_inst = self.unpack(data, 'QQ')
            generic = self.at(type_ptr, depth+1)
            count, args = self.unpack(class_inst, 'QQ')
            if count > 64:
                raise ValueError('invalid generic argument count')
            names = [self.at(p,depth+1)['name'] for p in self.unpack(args,f'{count}Q')]
            name = generic['name']+'<'+', '.join(names)+'>'
            base = generic['definition_index']
        elif kind in (19,30):
            name = ('!' if kind == 19 else '!!')+str(data)
        elif kind == 20:
            element, rank = self.unpack(data, 'QB')
            name = self.at(element,depth+1)['name']+'['+','*max(0,rank-1)+']'
        else:
            raise ValueError(f'unsupported native type tag {kind:#x}')
        result = dict(name=name+('&' if bits >> 29 & 1 else ''), attributes=bits & 65535,
                      kind=kind, definition_index=base)
        self.cache[va] = result
        return result

    def get(self, index):
        if not 0 <= index < len(self.pointers):
            raise ValueError('invalid native type index')
        return self.at(self.pointers[index])

    def enrich(self):
        offset_ptrs = self.unpack(self.registration['field_offsets_va'], f'{len(self.metadata.types)}Q')
        size_ptrs = self.unpack(self.registration['type_sizes_va'], f'{len(self.metadata.types)}Q')
        for typ in self.metadata.types:
            # Validate every byval type, independently of the search anchor counts.
            definition = self.get(typ['byval_type_index'])
            if definition['name'] != typ['name']:
                raise ValueError('native/metadata byval name mismatch')
            if definition['definition_index'] is not None and definition['definition_index'] != typ['index']:
                raise ValueError('native/metadata type identity mismatch')
            parent = typ['parent_type_index']
            if parent != (1 << (8*self.metadata.widths['type']))-1 and parent >= 0:
                parent_info = self.get(parent)
                typ['parent_type_name'] = parent_info['name']
                typ['parent_definition_index'] = parent_info['definition_index']
            fields = typ['fields']
            offsets = self.unpack(offset_ptrs[typ['index']], f'{len(fields)}i') if fields else []
            instance, native, static, thread_static = self.unpack(size_ptrs[typ['index']], 'IiII')
            typ['native_sizes'] = dict(instance=instance, native=native, static=static, thread_static=thread_static)
            for field, offset in zip(fields, offsets):
                info = self.get(field['type_index'])
                field['type_name'] = info['name']
                field['type_definition_index'] = info['definition_index']
                field['attributes'] = info['attributes']
                field['is_static'] = bool(info['attributes'] & 16)
                field['is_literal'] = bool(info['attributes'] & 64)
                field['offset_from_native_table'] = offset
            for method in typ['methods']:
                method['return_type_name'] = self.get(method['return_type_index'])['name']
