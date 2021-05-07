import re
import tarfile
from typing import Tuple, Iterator

from ase import Atoms


class ParserError(Exception):
    """Error raised when an occurs while parsing GDB9 dataset"""


_coord_line = (br'(?P<element>\D+)\s+(?P<x>-?\d+\.\d*(E-?\d+)?)\s+(?P<y>-?\d+\.\d*(E-?\d+)?)\s+'
               br'(?P<z>-?\d+\.\d*(E-?\d+)?)\s+(?P<pcharge>-?\d+\.\d*(E-?\d+)?)\s*')
_coord_re = re.compile(_coord_line)
_data_re = re.compile(
    br'^(?P<num_atoms>\d+)\n'
    br'gdb (?P<id>\d+)\s+(?P<A>-?\d+(\.\d*)?)\s+(?P<B>-?\d+\.\d*)\s+(?P<C>-?\d+\.\d*)\s+(?P<mu>-?\d+\.\d*)'
    br'\s+(?P<alpha>-?\d+\.\d*)\s+(?P<homo>-?\d+\.\d*)\s+(?P<lumo>-?\d+\.\d*)\s+(?P<gap>-?\d+\.\d*)'
    br'\s+(?P<r2>-?\d+\.\d*)\s+(?P<zpve>-?\d+\.\d*)\s+(?P<u_0>-?\d+\.\d*)\s+(?P<u_t>-?\d+\.\d*)'
    br'\s+(?P<h>-?\d+\.\d*)\s+(?P<g>-?\d+\.\d*)\s+(?P<cv>-?\d+\.\d*)\s+'
    br'(?P<coordinates>(' + _coord_line + br')+)'
    br'(?P<vib_freqs>(\s*-?\d+\.\d*)+)'
    br'(?P<smiles_gdb17>(\s*\S+))'
    br'(?P<smiles_opt>(\s*\S+))'
    br'(?P<inchi_corina>(\s*\S+))'
    br'(?P<inchi_opt>(\s*\S+){2})\s*$')


def parse_entry(string: bytes) -> Tuple[str, Atoms, dict]:
    elements = []
    positions = []

    match = _data_re.match(string)
    try:
        if not match:
            raise ParserError('String does not match pattern')

        for coord in _coord_re.finditer(match.group('coordinates')):
            elements.append(coord.group('element').decode('ascii').strip())
            positions.append((float(coord.group('x')), float(coord.group('y')), float(coord.group('z'))))

        info = {'smiles': match.group('smiles_opt').decode('ascii').strip()}

        return match.group('id').decode('ascii'), Atoms(symbols=elements, positions=positions), info

    except (ValueError, AttributeError) as e:
        raise ParserError(e)


def parse_dataset(file_path: str, strict=False) -> Iterator[Tuple[str, Atoms, dict]]:
    with tarfile.open(file_path, mode='r') as archive:
        for i, entry in enumerate(archive):
            f = archive.extractfile(entry)

            if not f:
                raise RuntimeError('File cannot be read')

            string = f.read().replace(b'*^', b'E')

            try:
                yield parse_entry(string)
            except ParserError as e:
                if not strict:
                    print('Could not parse: ' + entry.name + ': ' + str(e))
                else:
                    raise
