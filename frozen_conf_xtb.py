#!/usr/bin/env python

import sys
import scipy.spatial
import os
import random
import time
import datetime
import argparse
from shutil import copy2
import csv
import numpy as np
import timeit
from openbabel import openbabel
import itertools
from xtb.interface import Calculator, Param
from xtb.utils import get_solvent
from xtb.libxtb import VERBOSITY_MUTED
from xtb.interface import Environment
env = Environment()
env.set_output("error.log")
env.set_verbosity(VERBOSITY_MUTED)


# import pyXTB

ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
openbabel.obErrorLog.SetOutputLevel(0)

# What OS is being run?
if 'win' in sys.platform:
    print('This program does not run on Windows!\n')
    devnull = ''
    sys.exit(1)
else:   
    devnull = '>/dev/null'


N_to_element = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 
                9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 
                17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 
                25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 
                33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 
                41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 
                49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 
                57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 
                65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 
                73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 
                81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 
                89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 
                97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr", 104: "Rf", 
                105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn", 
                113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

element_to_N = {v: k for k, v in N_to_element.items()}


# Getting the timestamp
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d_%H%M%S')



"""
File management functions
"""


def fileToList(filename, split=False):
    """
    Input: name of a file that has data in a list,
       must have newline after last line
    Output: clean list with file contents
    """
    list_from_file = open(filename, 'r').readlines()
    if list_from_file[-1] == []:
        list_from_file = list_from_file[:-1]

    if split is True:
        list_from_file = [x.split() for x in list_from_file]

    return list_from_file


def analyze_infile(infile):
    """
    Takes in the input ZMAT or XYZ file, and spits out the structure as lists
    representing both XYZ and ZMAT formats.
    It uses OpenBabel to convert the data from ZMAT to XYZ and vise-versa.
    The ZMAT list does not have lines split.
    XYZ list has lines split, but all of the data is of type str.
    """
    # Getting the original values
    if '.xyz' in infile:
        filename_xyz = infile
        start_xyz = fileToList(infile, split=True)
        start_xyz = [[x[0]] + [float(y) for y in x[1:]] for x in start_xyz[2:]]
        filename_zmat = infile.replace('.xyz', '.zmat')
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "gzmat")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, infile)
        outZMAT = obConversion.WriteString(mol)
        obConversion.WriteFile(mol, filename_zmat)
        start_zmat = outZMAT.split('\n')[:-1]

    elif '.zmat' in infile:
        filename_zmat = infile
        filename_xyz = filename_zmat.replace('.zmat', '.xyz')
        start_zmat = fileToList(filename_zmat)
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("gzmat", "xyz")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, filename_zmat)
        outXYZ = obConversion.WriteString(mol)
        obConversion.WriteFile(mol, filename_xyz)
        outXYZ = outXYZ.split('\n')
        start_xyz = []
        for line in outXYZ[2:-1]:
            lspl = line.split()
            start_xyz.append([lspl[0]] + [float(y) for y in lspl[1:]])
    else:
        print("This program currently only works with xyz and gzmat files!\n"
              "Exiting...")
        sys.exit(1)
    return filename_xyz, filename_zmat, start_xyz, start_zmat


def print_out_file(data, save_filename, header=[], footer=[],
                   gen_xyz_line=False):
    """
    Dumps contents of 'data' into a file called 'save_filename'.
    Takes in additional options for adding headers and footers
       from different sources,
    and generating properly formatted XYZfile lines.
    """
    with open(save_filename, 'w') as file:
        if header != []:
            for h in header:
                file.write(h+'\n')
        for l in data:
            if gen_xyz_line is False:
                file.write(l+'\n')
            elif gen_xyz_line is True:
                file.write(generate_xyz_line(l))
        if footer != []:
            for f in footer:
                file.write(f+'\n')
    file.close()


def dump_xyzfile(geometry, filename, comment=''):
    with open(filename, 'w') as file:
        file.write(f"{len(geometry)}\n")
        file.write(f"{comment}\n")
        for line in geometry:
            file.write(generate_xyz_line(line))


def generate_xyz_line(record_list):
    '''
    This function writes down the XYZfile lines
    Based on the 'generate_pdb_line' function by Stacey Gerben
    '''
    element_symbol = "{:<3}".format(record_list[0])
    coord_x = "{:15.5f}".format(float(record_list[1]))
    coord_y = "{:15.5f}".format(float(record_list[2]))
    coord_z = "{:15.5f}".format(float(record_list[3]))

    entry = element_symbol + coord_x + coord_y + coord_z + "\n"

    return entry


"""
Conversion functions
"""


def xyzM_2_zmatM(xyz, split=False):
    """
    Converts an XYZ list, stored in memory,
       to ZMAT list that is stored in memory.
    Incoming XYZ list should have all items on all lines split,
       but coordinates stored as strings.
    Outgoing ZMAT has all lines split
    """
    n_atoms = len(xyz)
    xyz_strlines = []
    for l in xyz:
        xyz_strlines.append("         ".join(l))
    xyz_strline = '\n'.join(xyz_strlines)
    xyz_strline = str(n_atoms) + '\n' + 'asd\n' + xyz_strline + '\n'
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "gzmat")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz_strline)
    zmat = obConversion.WriteString(mol)
    zmat = zmat.split('\n')
    if split is False:
        return zmat
    elif split is True:
        outzmat = []
        for line in zmat:
            outzmat.append(line.split())
        return outzmat
    else:
        print('You\'re trying to do something weird '
              'with the xyzM_2_zmatM function.')
        sys.exit(1)


def zmatM_2_xyzM(zmat, split=False):
    """
    Converts a ZMAT list, stored in memory,
       to XYZ list that is stored in memory
    Incoming ZMAT list should have all items on all lines split,
       but all items stored as strings.
    Outgoing XYZ has all lines split but all items stored as strings.
    """
    var_text_line_no = zmat.index(['Variables:'])
    zmat_strlines = []
    for l in zmat[:var_text_line_no]:
        zmat_strlines.append("  ".join(l))
    for l in zmat[var_text_line_no:-1]:
        zmat_strlines.append(" ".join(l))
    zmat_strline = '\n'.join(zmat_strlines) + '\n'
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("gzmat", "xyz")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, zmat_strline)
    xyz = obConversion.WriteString(mol)
    xyz = xyz.split('\n')[2:]
    if split is False:
        return xyz
    elif split is True:
        outxyz = []
        for line in xyz:
            lspl = line.split()
            if len(lspl) < 4:
                continue
            outxyz.append([lspl[0]] + [float(y) for y in lspl[1:]])
        return outxyz
    else:
        print('You\'re trying to do something weird '
              'with the zmatM_2_xyzM function.')
        sys.exit(1)


"""
Calculation functions
"""
def get_dihedral(p):
    """
    Praxeolitic formula from Stackexchange:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python#
    1 sqrt, 1 cross product
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def xyzlist_2_xyzstr(xyzlist):
    xyzstr = ""
    if isinstance(xyzlist[0], list) and len(xyzlist[0]) == 4:
        xyzstr += "{}\ntempgeom\n".format(len(xyzlist))
    for l in xyzlist:
        xyzstr += generate_xyz_line(l)
    return xyzstr


def calculate_XTB_energy(geometry, charge, multiplicity=None, solvent='none'):
    
    atomnumbers = np.array([element_to_N[l[0]] for l in geometry])
    atompositions = np.array([np.array(x[1:]) for x in geometry])

    atompositions = atompositions * 1.8897259886  # converting Angstrom to Bohr

    # Executing XTB through Python interface without ASE
    calc = Calculator(Param.GFN2xTB, atomnumbers, atompositions, charge=charge)
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_solvent(get_solvent("h2o"))
    res = calc.singlepoint()

    return res.get_energy()


def get_smallest_interatomic_distance(xyz):
    xyz_noAtom = [row[1:] for row in xyz]
    distmat = scipy.spatial.distance.pdist(xyz_noAtom, 'euclidean')
    mindist = round(min(distmat), 3)
    return mindist, distmat


def gen_dihedrals(n_iter, start_dihedrals, dihedral_labels,
                  periodicities, constrained_dihedrals, mode='random'):
    """
    Input:
        n_iter: int :: how many dihedrals will be generated
        start_dihedrals: list(float) :: dihedral angle values from the starting geometry
        dihedral_labels: list(str) :: list of dihedral labels d##
        periodicities: list(float) :: list of rotation periods for each dihedral
        constrained_dihedrals: list(str) :: list of dihedral labels that should be constrained. The constraint value is specified in the 'periodicities' list.
        mode: 'random' or 'systematic' :: using either random number generator or systematic generation of dihedral angle values for each search step
    Output:
        dihedrals: list(list(float)) :: a list of sets of dihedrals that are considered at each search step. Length = n_iter.
    """
    dihedrals = []
    if mode == 'random':
        for n in range(n_iter):
            d_line = []
            for d, p in zip(dihedral_labels, periodicities):
                if d in constrained_dihedrals:
                    di = round(start_dihedrals[dihedral_labels.index(d)] +
                               float(random.randint(-1*p, p)), 1)
                else:
                    di = float(random.randint(-180/int(360.0/p),
                                              180/int(360.0/p)))
                d_line.append(di)
            dihedrals.append(d_line)
    elif mode == 'systematic':
        d_steps = []
        for d, p in zip(start_dihedrals, periodicities):
            n_steps = int(p / std)
            if n_steps < 2:
                # if the number of steps is too small then this will
                # guarantee that at least 5 dihedrals are generated
                steps = n_steps + 2
            elif n_steps == 2:
                steps = n_steps + 1
            elif n_steps < 5:
                steps = n_steps
            else:
                steps = int(n_steps/2 + 1)
            line = list(np.linspace(d-p/2, d, num=steps)) +\
                list(np.linspace(d, d+p/2, num=steps))

            for n in range(len(line)):
                if line[n] > 180.0:
                    line[n] = line[n] - 360.0
                elif line[n] < -180.0:
                    line[n] = line[n] + 360.0
            line = [round(x, 1) for x in line]
            unique_line = []
            for i in line:
                if i not in unique_line:
                    unique_line.append(i)
            if p < 360.0:
                unique_line = unique_line[:-1]
            print(unique_line)
            d_steps.append(unique_line)
        dihedrals = list(itertools.product(*d_steps))
        for i in dihedrals:
            if list(i) == start_dihedrals:
                dihedrals.remove(i)
                break
        print("{} combinations of dihedrals was generated "
              "for the search.".format(len(dihedrals)))
    return dihedrals


"""
Geometry parsing functions
"""


def get_constrained_dihedrals(constraints, dihedrals, pairs=[]):
    """
    Input:
        constraints: list(str or int) :: either dihedral labels d## or atom-pairs of dihedrals 
            that should have their range of considered torsion angles constrained to a specific value.
        dihedrals: list(str) :: a list of dihedral labels that are to be rotated
        pairs: list(int) :: a list of atom pairs that make up the dihedrals
            in the 'dihedrals' list.
    Output:
        constrained_dihedrals : list(str) :: a list of dihedral labels
    """
    constrained_dihedrals = []
    if constraints is not None:
        if all(x[0] == 'd' for x in constraints):
            # If the constraints input is already a list of dihedral labels
            # then just return that same list.
            constrained_dihedrals = constraints
        else:
            # If the input is a list of atom-pairs,
            # then figure out which dihedral label these atom-pairs belong to.
            # Return a list of these corresponding dihedral labels.
            constr_atoms = constraints
            n_pairs = int(len(constr_atoms)/2)
            constr_pairs = []
            for i in range(n_pairs):
                line = []
                line.append(constr_atoms[i*2])
                line.append(constr_atoms[1+i*2])
                constr_pairs.append(line)
            for p in constr_pairs:
                constrained_dihedrals.append(dihedrals[pairs.index(p)])
    return constrained_dihedrals


def setup_geometry(parameters, parameter_type,
                   zmat, xyz, manual_dihedral_reassign=False):
    """
    Input:
        parameters: list(int or str) :: a list of atom pairs or dihedral labels (d##)
        parameter_type: 'atoms' or 'dihedrals' :: the type of parameters in the 'parameters' list
        zmat: list(str) :: structure in ZMAT format, as an internally stored list; lines are of type str
        zyz: list(list(str)) :: structure in XYZ format, as an internally stored list; lines are split, but of type str
        manual_dihedral_reassign: bool :: True = User will be prompted to enter new 4th atom for some dihedrals;
                                          False = The program tries to figure out new 4th atoms by itself.
    Output:
        dihedrals: list(str) :: dihedral labels to be used in the conformer search
        zmat: list(str) :: modified ZMAT
        rot_di_labels: list(str) :: a list of labels of dihedrals used in the conformer search; format 'a1-a2-a3-a4'
        pairs: list(int) :: a list of integers where each subsequent pair corresponds to each dihedral label in the list 'dihedrals'

    Dihedral_Reassign :: this process checks if the user-requested atompairs
    belong to more than one dihedral definition in the ZMAT. This is to avoid
    structural distortions if more than one dihedrals share the same three base
    atoms, and only one is changed during conformational search.
    e.g. in the example below:
      A       E
       \     /
        B---C
             \
              D
    If the pair B-C belongs to two dihedral definitions D-C-B-A and E-C-B-A,
    then by default it is assumed that D-C-B-A is the correct one and will be
    used for conformer search.
    With manual_dihedral_reassign=True the user will be prompted to specify
    which one should be used for the search.
    E-C-B-A will be redefined so that it is no longer based on atom A.
    By default atom D is picked for that, giving a new definition E-C-B-D.
    With manual_dihedral_reassign=True the user will be prompted to specify
    which atom should be used to redefine that dihedral.
    New dihedral angle value for E-C-B-D is calculated after these steps.
    """

    var_text_line_no = zmat.index('Variables:')
    if parameter_type == 'atoms':
        if ',' in parameters:
            parameters = parameters.split(',')
        else:
            parameters = parameters.split()
        n_pairs = len(parameters)
        pairs = parameters.copy()

        for n in range(n_pairs):
            pairs[n] = pairs[n].split('-')

    # Extracting the dihedral identifiers from the zmat file
    # If multiple identifiers have been found for one central bond then the user
    # will be prompted for a new 4th atom to define that dihedral.
    # The program will calculate a new dihedral and modify the zmat file.
        dihedrals = []
        rot_di_labels = []
        for p in pairs:
            dis = []
            di_labels = []
            for line in zmat:
                r1 = '  ' + p[0] + '  r'
                a2 = '  ' + p[1] + '  a'
                r2 = '  ' + p[1] + '  r'
                a1 = '  ' + p[0] + '  a'

                if len(line.split()) > 5 and ((r1 in line and a2 in line) or
                                              (r2 in line and a1 in line)):
                    dis.append(line.split()[6])
                    di_labels.append(line.split())

            print("Dihedral angle(s) found for atoms {0} : {1}".format(p, dis))
            if len(dis) > 1:
                print("More than one dihedral is using atoms"
                      " {0} and {1} for the central bond!".format(p[0], p[1]))

                if manual_dihedral_reassign is True:
                    saved_dihedral = None
                    while saved_dihedral is None:
                        saved_dihedral =\
                         input("Which dihedral would you like to keep the same?"
                               "\nEnter the dihedral identifier 'd##' : ")
                        if 'd' not in saved_dihedral:
                            print("Please write dihedral name as 'd##'. "
                                  "Try again!")
                            saved_dihedral = None
                        elif saved_dihedral not in dis:
                            print("Please enter a dihedral name that is "
                                  "in the list: {} Try again!".format(dis))
                            saved_dihedral = None
                        else:
                            # User entered correct-looking data, all good!
                            continue

                elif manual_dihedral_reassign is False:
                    print("{0} was automatically chosen.\n"
                          "If this is not correct, use --man_di_reassign"
                          " flag when running the program to "
                          "choose it yourself.".format(dis[0]))
                    saved_dihedral = dis[0]

                dis_new = dis.copy()
                di_labels_new = di_labels.copy()
                di_labels_new.remove(di_labels[dis.index(saved_dihedral)])
                rdl = di_labels[dis.index(saved_dihedral)]
                rot_di_labels.append(rdl[-1][1:] + '-' + rdl[1] +
                                     '-' + rdl[3] + '-' + rdl[5])
                dis_new.remove(saved_dihedral)

                for n, m in zip(dis_new, di_labels_new):
                    di_label = n[1:] + '-' + m[1] + '-' + m[3] + '-' + m[5]

                    if manual_dihedral_reassign is True:
                        while True:
                            try:
                                new_4th_atom =\
                                 int(input("Please redefine the dihedral angle"
                                           " for atom {0}.\nEnter a new 4th"
                                           " atom number for the dihedral "
                                           "{1} : ".format(n[1:], di_label)))
                            except ValueError:
                                print('Please enter an integer!')
                                continue
                            if int(new_4th_atom) > int(n[1:]):
                                print("You can't define a dihedral through an"
                                      " atom that appears later in the "
                                      "atomlist. Try again!")
                                continue
                            elif int(new_4th_atom) == int(n[1:]):
                                print("You can't have the same first and "
                                      "fourth atom when defining a dihedral. "
                                      "Try again!")
                            else:
                                # An integer was entered, everyone's happy!
                                break
                    elif manual_dihedral_reassign is False:
                        print("{0} was automatically picked as the 4th atom "
                              "for dihedral {1}".format(saved_dihedral[1:],
                                                        di_label))
                        new_4th_atom = saved_dihedral[1:]

                    m_new = m.copy()
                    m_new[5] = str(new_4th_atom)
                    zmat[zmat.index('  '.join(m))] = "  ".join(m_new)

                    a1 = np.array(np.float_(xyz[int(n[1:])-1][1:]))
                    a2 = np.array(np.float_(xyz[int(m[1])-1][1:]))
                    a3 = np.array(np.float_(xyz[int(m[3])-1][1:]))
                    a4 = np.array(np.float_(xyz[int(new_4th_atom)-1][1:]))
                    new_dihedral = round(get_dihedral([a1, a2, a3, a4]), 3)

                    for l in zmat[var_text_line_no:-1]:
                        if n+'=' in l:
                            print("{0} will be defined through atom {1} "
                                  "with d = {2}".format(n, new_4th_atom,
                                                        new_dihedral))
                            lspl = l.split()
                            lspl[1] = str(new_dihedral)
                            zmat[zmat.index(l)] = " ".join(lspl)
            else:
                saved_dihedral = dis[0]
                rdl = di_labels[dis.index(saved_dihedral)]
                rot_di_labels.append(rdl[-1][1:] + '-' + rdl[1]
                                     + '-' + rdl[3] + '-' + rdl[5])
            dihedrals.append(saved_dihedral)

    elif parameter_type == 'dihedrals':
        # Only getting the atom labels for dihedrals
        # TODO: the code should also check if any other dihedral is using the
        # same central atompair as the user-entered dihedral
        rot_di_labels = []
        pairs = []
        dihedrals = parameters.copy()
        for di in dihedrals:
            for line in zmat:
                if len(line.split()) > 5 and line.split()[6] == di:
                    rdl = line.split()
                    rot_di_atoms = [rdl[-1][1:], rdl[1], rdl[3], rdl[5]]
                    rot_di_labels.append('-'.join(rot_di_atoms))
                    pairs.append([int(rdl[1]), int(rdl[3])])
                    break
    return dihedrals, zmat, rot_di_labels, pairs


def prompt_user_periodicities(rotable_dihedrals, rotable_dihedral_labels):
    print("Periodicities were not defined in the input, "
          "it is highly recommended that you do so.")
    print("Periodicity means the smallest rotation around a bond "
          "that would give the same structure.")
    print("For phenyl ring it would be 180 degrees, "
          "for CH3 it would be 120 degrees.")
    print("You can define them in the input with the '-p' flag: "
          "-p '360 360 180'")
    print("You will now be prompted for periodicities "
          "of each of the dihedrals:")
    print("If you enter \'all_360\' then all periodicities will"
          " be set to 360 degrees.\n")
    periodicities = []
    for d, l in zip(rotable_dihedrals, rotable_dihedral_labels):
        while True:
            try:
                prd = input("Please enter the periodicity for "
                            "{0} ({1}) : ".format(d, l))
                prd_test = float(prd)
            except ValueError:
                if prd == 'all_360':
                    while len(periodicities) < len(rotable_dihedrals):
                        periodicities.append(360.0)
                    return periodicities
                print('Please enter a number!')
                continue
            if abs(prd_test) > 360.0:
                print('Please enter a number between 0 and 360.')
                continue
            else:
                # All good!
                break
        periodicities.append(abs(prd_test))
    return periodicities


def get_rotable_dihedral_line_numbers(rotable_dihedrals, zmat):
    """
    Input:
        rotable_dihedrals: list(str) :: a list of dihedral labels used in
                                        the conformational search
        zmat: list(str) :: structure in ZMAT format
    Output:
        di_line_nos: list(int) :: line numbers in the ZMAT list where the
                                  corresponding dihedral values are defined.
    """
    var_text_line_no = zmat.index('Variables:')
    di_line_nos = []
    for d in rotable_dihedrals:
        for l in zmat[var_text_line_no:-1]:
            if d+'=' in l:
                di_line_nos.append(zmat.index(l))
    return di_line_nos


def compare_dihedral_similarity(dihedrals, old_dihedrals):
    """
    Input:
        dihedrals: list(float) :: dihedral angles that are considered in a
                                  conformational search step
        old_dihedrals: list(list(??,??,list(float))) :: log of conformation
                     search steps dihedral angles are the [2] item on each line
    Output:
        old_similar_dihedral: list(float) :: first matching already considered
                                             set dihedral angles

    Similarity is determined based on whether all dihedrals in the new set
    [d1,d2,d3] are within similarity threshold (std) of some considered set
    [od1,od2,od3] in a way that all(odN - std <= dN <= odN + std) == True.
    """
    for l in old_dihedrals:
        line = l[2]
        old_set = []
        new_set = []
        for n in range(0, len(dihedrals)):
            # If the generated dihedral is close to 180 or -180 degrees
            # then the comparison will be made in 360 degree space.
            if abs(dihedrals[n]) >= 180.0 - std:
                if dihedrals[n] < 0.0:
                    comp_angle = 360.0 + dihedrals[n]
                if dihedrals[n] > 0.0:
                    comp_angle = dihedrals[n]
                if line[n] < 0.0:
                    tstd_angle = 360.0 + line[n]
                if line[n] > 0.0:
                    tstd_angle = line[n]
            else:
                tstd_angle = line[n]
                comp_angle = dihedrals[n]
            old_set.append(tstd_angle)
            new_set.append(comp_angle)
        if all(x - std <= y <= x + std for x, y in zip(old_set, new_set)):
            # if some combination of dihedrals gives matches with something
            # then that dataline is returned
            return l
        else:
            continue
    # if no matches were found then None is returned
    return None


def plot_dihedrals(dihedrals, dihedral_labels, periodicities, constraints):
    """
    Testing function to plot the histogram of considered dihedrals.
    Is not called during normal program operation.
    """
    import matplotlib.pyplot as plt
    n_dis = len(dihedrals[0])
    plt.figure(figsize=(6, 6))
    for n in range(n_dis):
        plt.subplot(n_dis // 2 + 1, 2, n+1)
        data = [x[n] for x in dihedrals]
        plt.hist(data, bins=range(int(min(data)),
                                  int(max(data)) + int(std), int(std)),
                 range=[-180.0, 180.0], align='mid')
        plt.xlabel(dihedral_labels[n])
        plt.xticks([-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0])
        plt.tight_layout()


parser = argparse.ArgumentParser()

parser.add_argument("--xyz", type=str, required=True, help="Input XYZ file")
parser.add_argument("--atoms", type=str, nargs="+", help="Define atoms around which the bonds will be rotated (pairs X-Y and Z-W)")
parser.add_argument("--dihedrals", type=str, nargs="+", help="Can be used instead of --atoms when using ZMAT as input file. Specify dihedral names found in ZMAT file.")
parser.add_argument("--per", type=str, nargs="+", help="Periodicities of rotations around each bond. 180.0 means that after a 180-degree rotation you get the same structure.")
parser.add_argument("--constr", type=str, nargs="+", help="Specify which dihedral(s) should have its/their rotation(s) constrained to +- a value. Use --per to specify that value.")
parser.add_argument("--charge", type=int, default=0, help="(default = 0) Charge of the system.")
parser.add_argument("--multi", type=int, default=1, help="(default = 1) Multiplicity (# of unpaired electrons + 1) of the system.")
parser.add_argument("--systematic", default=False, action="store_true", help="Switches to systematic conformer generation. Likely to take much longer.")
parser.add_argument("--iter", type=int, default=100, help="(default = 100) How many iterations of random conformer generation the program will do. Ignored when doing a systematic scan.")
parser.add_argument("--save", type=int, default=10, help="(default = 10) How many different lowest energy conformers are saved as output.")
parser.add_argument("--similarity", type=float, default=20.0, help="(default = 20.0) Smallest allowed difference between all dihedrals of two conformers. OR\n        Step size when doing a systematic scan.")
parser.add_argument("--step", type=float, default=20.0, help="(default = 20.0) Smallest allowed difference between all dihedrals of two conformers. OR\n        Step size when doing a systematic scan.")
parser.add_argument("--man_di_reassign", default=False, action="store_true", help="Switches to systematic conformer generation. Likely to take much longer.")
parser.add_argument("--test", default=False, action="store_true", help="Switches to systematic conformer generation. Likely to take much longer.")
parser.add_argument("--solvent", type=str, help="Solvent used for GBSA energy calculations.")

args = parser.parse_args()


filename = args.xyz

params = args.atoms
param_type = 'atoms'
if args.atoms is None:
    assert args.dihedrals is not None
    params = args.dihedrals
    param_type = 'dihedrals'

man_di_reassign = args.man_di_reassign
charge = args.charge
multiplicity = args.multi
solvent = args.solvent
constraints = args.constr

periodicities = []
if args.per is not None:
    periodicities = args.per
if 'all_360' not in periodicities:
    periodicities = [float(x) for x in periodicities]

testmode = args.test

gen_mode = 'random'
if args.systematic is True:
    gen_mode = "systematic"

iterations = args.iterations
n_save = args.save

if args.similarity is not None:
    std = args.similarity
elif args.step is not None:
    std = args.step
else:
    sys.exit(1)

dihedralsfile = None

'''
END SETTINGS
'''

empirical = True
dl = None

# Analyzing the input file and producing an univeral output.
filename_xyz, filename_zmat, start_xyz, start_zmat =\
    analyze_infile(filename)

# Setting up the geometry, fixing the dihedral definitions in the ZMAT
# that use the same central bond.
dihedrals, start_zmat, rot_di_labels, pairs =\
    setup_geometry(params, param_type, start_zmat, start_xyz,
                   manual_dihedral_reassign=man_di_reassign)

# Getting the dihedral names that are to be constrained.
constr_dihedrals = get_constrained_dihedrals(constraints,
                                             dihedrals, pairs=pairs)

# Setting the directory name for all of the generated files.
WDIR = os.path.basename(filename) + '_' + timestamp

# If periodicities of dihedral rotations were not defined by the user
# in the arguments then they are prompted to enter them.
if periodicities == []:
    periodicities = prompt_user_periodicities(dihedrals, rot_di_labels)
elif periodicities == ['all_360']:
    periodicities = [360.0]*len(dihedrals)

# Getting the linenumbers in the ZMAT where
# the rotable dihedral labels are located.
dihedrals_line_nos = get_rotable_dihedral_line_numbers(dihedrals,
                                                       start_zmat)

# Printing out a modified ZMAT file
print_out_file(start_zmat, filename_zmat)

# Splitting up the lines in the starting ZMAT that is stored
# in the memory for future use.
start_zmat_splt = []
for line in start_zmat:
    start_zmat_splt.append(line.split())

# If user requested that run to be just a test,
# then this is the point where the program exits.
if testmode:
    print('User requested test mode, exiting...')
    sys.exit(1)

# Making a directory where all of the produced XYZ files are saved.
os.mkdir(WDIR)
copy2(filename_xyz, WDIR + '/' + filename_xyz.replace('.xyz', '_0.xyz'))

# sys.exit(0)

'''
START STARTING POINT ANALYSIS
'''

# Calculating initial energy

energy = calculate_XTB_energy(start_xyz, charge,
                              multiplicity=multiplicity,
                              solvent=solvent)

# Calculating the threshold for second filter:
# Smallest interatomic distance minus 1%
start_mindist, start_distmat = get_smallest_interatomic_distance(start_xyz)
start_mindist = round(start_mindist - start_mindist/100, 3)

# Getting the number of neighboring atoms for empirical clash filtering
EMP_D_LIMIT = 1.8
EMP_EXTRA_NEIGHBORS = 6
N_neighbors = len([x for x in start_distmat if x < EMP_D_LIMIT])

# Getting the starting dihedral values
start_dihedrals = []
for d in dihedrals_line_nos:
    start_dihedrals.append(float(start_zmat[d].split()[1]))
print(dihedrals)
print(start_dihedrals)

# Setting up lists that record data from analysis steps
# TODO: could these things be redefined to a dictionary?

# Saving all produced XYZ's in a list in memory
all_xyzs = []
all_xyzs.append(start_xyz)

# Energies and iteration numbers that pass tests,
# will be mutated throughout the process.
energies_n = []
energies_n.append([0, energy] + [start_dihedrals])

all_energies_n = []  # Energies and iteration numbers of all steps
all_energies_n.append([0, energy])

# Defining the ranking for the first iteration:
ranking = sorted(energies_n, key=lambda x: float(x[1]))[:n_save]

tested_dihedrals = []  # all dihedrals that are being generated
tested_dihedrals.append(start_dihedrals)

passed_dihedrals = []  # dihedrals that pass the clash test
passed_dihedrals.append(start_dihedrals)

# keeping track of everything that is being done, for the output log:
logged_data = [[0, start_dihedrals, energy, 'input geometry']]

'''
END STARTING POINT ANALYSIS
'''

'''
START CONFORMATIONAL ANALYSIS
'''

temp_zmat = start_zmat_splt.copy()

# dl = 'dihedrals.pickle'

# Generating a list of dihedral angle values that will be analyzed.
# The list is either generated randomly (faster) or systematically (slow)
if dihedralsfile is None:
    import pickle
    generated_dihedrals = gen_dihedrals(iterations, start_dihedrals, dihedrals,
                                        periodicities, constr_dihedrals,
                                        mode=gen_mode)
    tested_dihedrals = tested_dihedrals + generated_dihedrals
    with open('dihedrals.pickle', 'wb') as file:
        pickle.dump(tested_dihedrals, file)

else:
    import pickle
    with open(dihedralsfile, 'rb') as file:
        tested_dihedrals = pickle.load(file)

# sys.exit(0)
start = timeit.default_timer()

n_iter = 0
for d_line in tested_dihedrals[1:]:
    n_iter += 1

    filename_n_iter_xyz = filename_xyz.replace('.xyz',
                                               '_' + str(n_iter) + '.xyz')

    logged_data.append([n_iter, d_line])

    # Modifying the ZMAT in memory with the generated dihedrals
    for l, d in zip(dihedrals_line_nos, d_line):
        temp_zmat[l][1] = str(d)

    xyz = zmatM_2_xyzM(temp_zmat, split=True)
    all_xyzs.append(xyz)

    # Testing for clashes in the system.
    # XYZ of the new conformer is generated from the prevously modified ZMAT
    # Distance matrix is calculated
    # If min(distmat) is smaller than the smallest interatomic distance
    # in the start structure (start_mindist)
    # then that generated structure is discarded.
    mindist, distmat = get_smallest_interatomic_distance(xyz)

    if mindist < start_mindist:
        logged_data[n_iter] = logged_data[n_iter] +\
                              ['', 'Rejected due to clashes.']
        # print("# {} rejected due to too small interatomic distance(s)."
        #       " Potential clash detected!".format(n_iter))
        continue

    # Doing the additional empirical clash filtering
    # with the number of neighboring atoms

    if empirical:
        N_neighbors_new = len([x for x in distmat if x < EMP_D_LIMIT])
        if N_neighbors_new - N_neighbors > EMP_EXTRA_NEIGHBORS:
            logged_data[n_iter] = logged_data[n_iter] +\
                                  ['', 'Rejected due to clashes.']
            # print("# {} rejected due to too many close atom neighbors."
            #       " Potential clash detected! TEST".format(n_iter))
            continue

    passed_dihedrals.append(d_line)

    energy = calculate_XTB_energy(xyz, charge,
                                  multiplicity=multiplicity,
                                  solvent=solvent)

    # If XTB gave some error and didn't exit normally:
    if energy is None:
        logged_data[n_iter] = logged_data[n_iter] +\
                              ['', 'Energy calculation failed.']
        print("# {0} rejected because XTB energy"
              " calculation failed.".format(n_iter))
        continue

    deltaE = round((energy -
                    min([row[1] for row in all_energies_n]))*627.509, 2)
    all_energies_n.append([n_iter, energy, d_line])

    # Checking how the energy ranks and filtering out only top N energies
    if energy > ranking[-1][1] and len(ranking) == n_save:
        logged_data[n_iter] = logged_data[n_iter] +\
                              [energy, 'Rejected due to high energy.']
        print(f"# {n_iter} rejected due to too high energy. dE = {deltaE}")
        continue

    similar_dihedral = compare_dihedral_similarity(d_line, ranking)

    if similar_dihedral is not None:
        similar_dihedral_N = similar_dihedral[0]
        similar_dihedral_E = similar_dihedral[1]

    # If similar structure was found previously then the energies are compared.
    # If the new structure has lower energy then it will replace the old one.
        if energy < similar_dihedral_E:
            if similar_dihedral_N == 0:
                energies_n.append([n_iter, energy, d_line])
                logged_data[n_iter] = logged_data[n_iter] +\
                    [energy, f"Similar to #{similar_dihedral_N}, both kept."]
                dump_xyzfile(xyz,
                             filename=f"{WDIR}/{filename_n_iter_xyz}",
                             comment=f"{filename_n_iter_xyz} {energy}")
                print("# {0} has similar geometry to # {1}. "
                      "Lower energy found (dE = {2}), "
                      "both are kept".format(n_iter,
                                             similar_dihedral_N,
                                             deltaE))

            else:
                for row in energies_n:
                    if row[0] == similar_dihedral_N:
                        similar_fname = WDIR + '/' +\
                         filename_xyz.replace('.xyz', '_' +
                                              str(similar_dihedral_N) + '.xyz')
                        energies_n[energies_n.index(row)] = [n_iter, energy,
                                                             d_line]
                        logged_data[similar_dihedral_N] =\
                            logged_data[similar_dihedral_N] +\
                            ['Replaced by #' + str(n_iter)]
                        logged_data[n_iter] =\
                            logged_data[n_iter] +\
                            [energy, 'Replacing #' + str(similar_dihedral_N)]
                        print("# {0} has similar geometry to # {1}. Lower "
                              "energy found (dE = {2}), structure "
                              "replaced.".format(n_iter,
                                                 similar_dihedral_N, deltaE))
                        dump_xyzfile(xyz,
                                     filename=f"{WDIR}/{filename_n_iter_xyz}",
                                     comment=f"{filename_n_iter_xyz} {energy}")

                        os.remove(similar_fname)
        else:
            logged_data[n_iter] = logged_data[n_iter] +\
             [energy, 'Rejected due to similarity with #' +
              str(similar_dihedral_N) + ' and higher energy.']
            print("# {0} has similar geometry to # {1}. "
                  "Higher energy found, "
                  "no action taken.".format(n_iter, similar_dihedral_N))

    else:
        energies_n.append([n_iter, energy, d_line])
        logged_data[n_iter] = logged_data[n_iter] + [energy]
        print("# {0} saved! dE = {1} kcal/mol".format(n_iter, deltaE))
        dump_xyzfile(xyz,
                     filename=f"{WDIR}/{filename_n_iter_xyz}",
                     comment=f"{filename_n_iter_xyz} {energy}")

    ranking = sorted(energies_n, key=lambda x: float(x[1]))[:n_save]


'''
END CONFORMATIONAL ANALYSIS
'''

'''
START OUTPUT GENERATION
'''

for i in sorted(energies_n, key=lambda x: float(x[1]))[n_save:]:
    fname_delete = filename_xyz.replace('.xyz', '_'+str(i[0]) + '.xyz')
    if os.path.isfile(WDIR+'/'+fname_delete):
        os.remove(WDIR+'/'+fname_delete)

with open(WDIR+'/saved_xyz.xyz', 'w') as file:
    for s in ranking:
        dE = round((s[1] - min([row[1] for row in ranking]))*627.509, 2)
        s.append(dE)
        fname = filename_xyz.replace('.xyz', '_' + str(s[0]) + '.xyz')
        file.write(str(len(all_xyzs[s[0]]))+'\n')
        file.write(fname + ' ' + str(s[1])+'\n')
        for l in all_xyzs[s[0]]:
            file.write(generate_xyz_line(l))
        print(s)

print("{} conformers were tested.".format(len(tested_dihedrals)))
print("{} conformers passed the clash filter.".format(len(passed_dihedrals)))
print('Results have been saved into the directory: ' + WDIR)
print("All analyzed structures have been saved to 'all_xyz.xyz'")
print("Top " + str(n_save) + " ranked structures have been saved"
      " to 'saved_xyz.xyz'")

with open(WDIR+'/all_xyz.xyz', 'w') as file:
    for s, d in zip(all_xyzs, logged_data):
        file.write(str(len(s))+'\n')
        file.write(" ".join([str(x) for x in d])+'\n')
        for l in s:
            file.write(generate_xyz_line(l))

outfilename = WDIR + '/' + filename_xyz.replace('.xyz', '.log')
with open(outfilename, 'w') as outfile:
    outfile.write('N      ' + str(dihedrals) +
                  '        Energy            Comment' "\n")
    w = csv.writer(outfile, dialect='excel-tab')
    w.writerows(logged_data)

'''
END OUTPUT GENERATION
'''
stop = timeit.default_timer()
print("Program execution took {:.3f} seconds.".format((stop - start)))

# Doing some clean-up
XTB_files = ['charges', 'wbo', 'molden.input', 'xtbrestart']
for f in XTB_files:
    if os.path.exists(f):
        os.remove(f)
