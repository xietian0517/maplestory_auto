"""Offline IL2CPP v39 metadata analysis; file offsets are NOT memory offsets."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import struct

SECTIONS = '''stringLiteral stringLiteralData string events properties methods
parameterDefaultValues fieldDefaultValues fieldAndParameterDefaultValueData
fieldMarshaledSizes parameters fields genericParameters genericParameterConstraints
genericContainers nestedTypes interfaces vtableMethods interfaceOffsets typeDefinitions
images assemblies fieldRefs referencedAssemblies attributeData attributeDataRange
unresolvedVirtualCallParameterTypes unresolvedVirtualCallParameterRanges
windowsRuntimeTypeNames windowsRuntimeStrings exportedTypeDefinitions'''.split()


def index_width(count):
    # Max unsigned value is the null sentinel; count entries use 0..count-1.
    return 1 if count <= 255 else 2 if count <= 65535 else 4


class Metadata:
    def __init__(self, data):
        self.data = data
        if len(data) < 8 + 12 * len(SECTIONS):
            raise ValueError('truncated metadata header')
        magic, self.version = struct.unpack_from('<II', data)
        if magic != 0xFAB11BAF or self.version != 39:
            raise ValueError(f'expected unmodified IL2CPP v39 header, got {magic:#x} v{self.version}')
        self.sections = {}
        for index, name in enumerate(SECTIONS):
            offset, size, count = struct.unpack_from('<iii', data, 8+12*index)
            if min(offset, size, count) < 0 or offset+size > len(data):
                raise ValueError(f'invalid section: {name}')
            self.sections[name] = dict(offset=offset, size=size, count=count)

    def rows(self, section, fmt):
        info = self.sections[section]
        layout = struct.Struct('<'+fmt)
        if info['size'] != info['count']*layout.size:
            raise ValueError(f'{section} row size mismatch: expected {layout.size}')
        return [(info['offset']+i*layout.size, layout.unpack_from(self.data, info['offset']+i*layout.size))
                for i in range(info['count'])]

    def decode(self):
        # The type-index width is inferable from the field row's fixed 8 bytes.
        section = self.sections['fields']
        if not section['count'] or section['size'] % section['count']:
            raise ValueError('invalid fields table')
        type_width = section['size']//section['count']-8
        char = {1:'B', 2:'H', 4:'i'}
        def width(name):
            return index_width(self.sections[name]['count'])
        t, d, g, p = [char[w] for w in (type_width, width('typeDefinitions'), width('genericContainers'), width('parameters'))]
        self.widths = dict(type=type_width, definition=width('typeDefinitions'), generic=width('genericContainers'), parameter=width('parameters'))
        self.fields = [dict(index=i, metadata_offset=off, name=self.string(r[0]), type_index=r[1], token=r[2])
                       for i, (off,r) in enumerate(self.rows('fields', 'i'+t+'I'))]
        self.methods = [dict(index=i, metadata_offset=off, name=self.string(r[0]), declaring_type=r[1],
                             return_type_index=r[2], parameter_start=r[4], token=r[6], flags=r[7], parameter_count=r[10])
                        for i,(off,r) in enumerate(self.rows('methods', 'i'+d+t+'I'+p+g+'IHHHH'))]
        if any(f['token'] >> 24 != 4 for f in self.fields):
            raise ValueError('invalid field token')
        if any(m['token'] >> 24 != 6 for m in self.methods):
            raise ValueError('invalid method token')
        parameter_count = self.sections['parameters']['count']
        if any(m['parameter_count'] and not 0 <= m['parameter_start'] <= parameter_count-m['parameter_count'] for m in self.methods):
            raise ValueError('invalid method parameter range')
        self.types = []
        self.byval_names = {}
        for i, (off,r) in enumerate(self.rows('typeDefinitions', 'ii'+t*3+g+'I'+'i'*8+'H'*8+'II')):
            name, namespace = self.string(r[0]), self.string(r[1])
            full_name = namespace+'.'+name if namespace else name
            fs, ms, mc, fc = r[7], r[8], r[15], r[17]
            for start, count, table in [(fs,fc,self.fields),(ms,mc,self.methods)]:
                if count and (start < 0 or start+count > len(table)):
                    raise ValueError(f'invalid member range on {full_name}')
            fields = self.fields[fs:fs+fc] if fc else []
            methods = self.methods[ms:ms+mc] if mc else []
            if any(m['declaring_type'] != i for m in methods):
                raise ValueError(f'method owner mismatch: {full_name}')
            if r[24] >> 24 != 2:
                raise ValueError(f'invalid type token: {full_name}')
            self.byval_names[r[2]] = full_name
            self.types.append(dict(index=i, metadata_offset=off, name=full_name,
                                   byval_type_index=r[2], parent_type_index=r[4], token=r[24],
                                   fields=fields, methods=methods))
        self.images = []
        for off, r in self.rows('images', 'ii'+d+'I'+d+'IiI'+'iI'):
            name, start, count = self.string(r[0]), r[2], r[3]
            if count and start+count > len(self.types):
                raise ValueError('invalid image type range')
            self.images.append(dict(name=name, start=start, count=count))
            for typ in self.types[start:start+count]:
                if 'assembly' in typ:
                    raise ValueError('overlapping image type ranges')
                typ['assembly'] = name
        if any('assembly' not in t for t in self.types):
            raise ValueError('types missing assembly')
        for field in self.fields:
            # Generic/pointer/array types need the native type table; never guess.
            field['type_name'] = self.byval_names.get(field['type_index'])
        for typ in self.types:
            typ['parent_type_name'] = self.byval_names.get(typ['parent_type_index'])
        return self.types

    def string(self, index):
        section = self.sections['string']
        if not 0 <= index < section['size']:
            raise ValueError(f'invalid string index: {index}')
        start = section['offset']+index
        end = self.data.find(b'\0', start, section['offset']+section['size'])
        if end < 0:
            raise ValueError('unterminated metadata string')
        return self.data[start:end].decode('utf-8')

    def literals(self):
        indices = [r[0] for _, r in self.rows('stringLiteral','i')]
        section = self.sections['stringLiteralData']
        # v35+ stores one extra terminal offset instead of per-string lengths.
        if len(indices) != section['count']+1 or indices[-1] != section['size']:
            raise ValueError('invalid string literal terminal offset')
        result = []
        for i,(start,end) in enumerate(zip(indices,indices[1:])):
            if not 0 <= start <= end <= section['size']:
                raise ValueError('invalid string literal range')
            result.append(dict(index=i,metadata_offset=section['offset']+start,
                               value=self.data[section['offset']+start:section['offset']+end].decode('utf-8', errors='replace')))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('analysis/il2cpp'))
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    data = args.metadata.read_bytes()
    metadata = Metadata(data)
    types = metadata.decode()
    native_report = None
    if args.binary:
        from .pe_analysis import PE, NativeTypes
        binary = args.binary.read_bytes()
        native = NativeTypes(PE(binary), metadata)
        native.enrich()
        native_report = dict(file=str(args.binary), sha256=hashlib.sha256(binary).hexdigest(),
                             registration=native.registration, runtime_verified=False)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(file=str(args.metadata), sha256=hashlib.sha256(data).hexdigest(),
                  version=metadata.version, sections=metadata.sections, native=native_report)
    (args.output/'header.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.output/'types.json').write_text(json.dumps(types, ensure_ascii=False), encoding='utf-8')
    (args.output/'images.json').write_text(json.dumps(metadata.images, ensure_ascii=False, indent=2), encoding='utf-8')
    terms = re.compile(r'player|monster|mob(?:manager|pool|controller|object|entity)|foothold|ladder|(?:^|_)rope|mapmanager|fieldmanager|charactercontroller|localuser', re.I)
    game = [t for t in types if t['assembly'] in ('Assembly-CSharp.dll', 'Framework.dll')]
    candidates = [t for t in game if terms.search(t['name'])]
    (args.output/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
    literal_terms = re.compile(r'foothold|ladder|monster|mob/|mobid|playerpos|mapid|localplayer|\brope\b|walkspeed|jumpheight', re.I)
    literals = [s for s in metadata.literals() if len(s['value']) < 400 and literal_terms.search(s['value'])]
    (args.output/'literal-clues.json').write_text(json.dumps(literals,ensure_ascii=False,indent=2),encoding='utf-8')
    if native_report:
        from .metadata_report import export_findings
        export_findings(metadata, args.output, report, literals)
    print(json.dumps(dict(version=metadata.version, widths=metadata.widths, types=len(types),
                          native=native_report, game_name_matches=len(candidates)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
