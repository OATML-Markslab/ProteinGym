from Bio.PDB import PDBParser, PDBIO, Select


class ResidueSelect(Select):
    def __init__(self, residue_numbers):
        self.residue_numbers = residue_numbers

    def accept_residue(self, residue):
        return residue.id[1] in self.residue_numbers


def filter_residues(input_pdb_file, output_pdb_file, residue_numbers):
    parser = PDBParser()
    structure = parser.get_structure("structure", input_pdb_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file, ResidueSelect(residue_numbers))
