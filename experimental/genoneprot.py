import argparse
import os
import re
from pathlib import Path

import torch
from esm import pretrained
from esm.inverse_folding.util import load_coords, get_encoder_output


def _setup_residue_map() -> None:
	try:
		from biotite.sequence import ProteinSequence

		m = ProteinSequence._dict_3to1
		canon_map = {
			"SER": "S", "DSN": "S",
			"SEP": "S", "TPO": "T", "PTR": "Y",
			"MSE": "M", "SEC": "C", "PYL": "K",
			"CSO": "C", "CSD": "C", "CME": "C", "CYX": "C", "CSS": "C",
			"HIP": "H", "HIE": "H", "HID": "H",
			"ASH": "D", "GLH": "E",
			"MHO": "M", "HYP": "P",
			"ACE": "X", "NME": "X",
		}
		for k, v in canon_map.items():
			m.setdefault(k, v)

		substitutions_3to3 = {
			"2AS":"ASP", "3AH":"HIS", "5HP":"GLU", "ACL":"ARG", "AGM":"ARG", "AIB":"ALA", "ALM":"ALA", "ALO":"THR", "ALY":"LYS", "ARM":"ARG",
			"ASA":"ASP", "ASB":"ASP", "ASK":"ASP", "ASL":"ASP", "ASQ":"ASP", "AYA":"ALA", "BCS":"CYS", "BHD":"ASP", "BMT":"THR", "BNN":"ALA",
			"BUC":"CYS", "BUG":"LEU", "C5C":"CYS", "C6C":"CYS", "CAS":"CYS", "CCS":"CYS", "CEA":"CYS", "CGU":"GLU", "CHG":"ALA", "CLE":"LEU", "CME":"CYS",
			"CSD":"ALA", "CSO":"CYS", "CSP":"CYS", "CSS":"CYS", "CSW":"CYS", "CSX":"CYS", "CXM":"MET", "CY1":"CYS", "CY3":"CYS", "CYG":"CYS",
			"CYM":"CYS", "CYQ":"CYS", "DAH":"PHE", "DAL":"ALA", "DAR":"ARG", "DAS":"ASP", "DCY":"CYS", "DGL":"GLU", "DGN":"GLN", "DHA":"ALA",
			"DHI":"HIS", "DIL":"ILE", "DIV":"VAL", "DLE":"LEU", "DLY":"LYS", "DNP":"ALA", "DPN":"PHE", "DPR":"PRO", "DSN":"SER", "DSP":"ASP",
			"DTH":"THR", "DTR":"TRP", "DTY":"TYR", "DVA":"VAL", "EFC":"CYS", "FLA":"ALA", "FME":"MET", "GGL":"GLU", "GL3":"GLY", "GLZ":"GLY",
			"GMA":"GLU", "GSC":"GLY", "HAC":"ALA", "HAR":"ARG", "HIC":"HIS", "HIP":"HIS", "HMR":"ARG", "HPQ":"PHE", "HTR":"TRP", "HYP":"PRO",
			"IAS":"ASP", "IIL":"ILE", "IYR":"TYR", "KCX":"LYS", "LLP":"LYS", "LLY":"LYS", "LTR":"TRP", "LYM":"LYS", "LYZ":"LYS", "MAA":"ALA", "MEN":"ASN",
			"MHS":"HIS", "MIS":"SER", "MLE":"LEU", "MPQ":"GLY", "MSA":"GLY", "MSE":"MET", "MVA":"VAL", "NEM":"HIS", "NEP":"HIS", "NLE":"LEU",
			"NLN":"LEU", "NLP":"LEU", "NMC":"GLY", "OAS":"SER", "OCS":"CYS", "OMT":"MET", "PAQ":"TYR", "PCA":"GLU", "PEC":"CYS", "PHI":"PHE",
			"PHL":"PHE", "PR3":"CYS", "PRR":"ALA", "PTR":"TYR", "PYX":"CYS", "SAC":"SER", "SAR":"GLY", "SCH":"CYS", "SCS":"CYS", "SCY":"CYS",
			"SEL":"SER", "SEP":"SER", "SET":"SER", "SHC":"CYS", "SHR":"LYS", "SMC":"CYS", "SOC":"CYS", "STY":"TYR", "SVA":"SER", "TIH":"ALA",
			"TPL":"TRP", "TPO":"THR", "TPQ":"ALA", "TRG":"LYS", "TRO":"TRP", "TYB":"TYR", "TYI":"TYR", "TYQ":"TYR", "TYS":"TYR", "TYY":"TYR",
		}

		std_3to1 = {
			"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
			"HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
			"THR":"T","TRP":"W","TYR":"Y","VAL":"V",
		}

		for k3, v3 in substitutions_3to3.items():
			v1 = std_3to1.get(v3.upper())
			if v1 and k3 not in m:
				m[k3] = v1
	except Exception:
		pass


def _infer_chain(struct_path: Path) -> str:
	name = struct_path.name
	if struct_path.suffix == ".cif" and "AF-" in name:
		return "A"
	match = re.search(r"_(?P<chain>[A-Za-z0-9]+)\.(?:cif|pdb)$", name)
	if match:
		return match.group("chain")
	return "A"


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--cif", type=str, required=True, help="Path to CIF or PDB file")
	parser.add_argument("--out", type=str, default=None, help="Output .pt path")
	parser.add_argument("--chain", type=str, default=None, help="Chain ID (default: inferred)")
	parser.add_argument("--device", type=str, default=None)
	args = parser.parse_args()

	struct_path = Path(args.cif)
	if not struct_path.exists():
		raise FileNotFoundError(f"Structure not found: {struct_path}")

	_setup_residue_map()

	device = args.device
	if device is None:
		if torch.backends.mps.is_available():
			device = "mps"
		elif torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"

	model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()
	model = model.to(device).eval()

	chain = args.chain or _infer_chain(struct_path)
	coords, seq = load_coords(str(struct_path), chain=chain)

	if "X" in seq:
		print("[warn] sequence contains 'X' (unknown residues)")

	rep = get_encoder_output(model, alphabet, coords)
	rep = rep.detach().cpu()

	out_path = Path(args.out) if args.out else struct_path.with_suffix(struct_path.suffix + ".pt")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(rep, out_path)

	print(f"Saved embedding: {out_path}")
	print(f"Shape: {tuple(rep.shape)}")


if __name__ == "__main__":
	main()
