#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import numpy as np

from nomad.datamodel import EntryArchive
from fhivibesparser.fhivibes_parser import FHIVibesParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return FHIVibesParser()


def test_singlepoint(parser):
    archive = EntryArchive()
    parser.parse('tests/data/singlepoint.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'single_point'

    sec_run = archive.section_run
    assert len(sec_run) == 10
    assert len(sec_run[2].section_single_configuration_calculation) == 1

    assert sec_run[8].section_system[0].atom_positions[3][2].magnitude == approx(5.30098546e-10)
    assert sec_run[5].section_system[0].atom_velocities[1][0].magnitude == approx(-2.18864066e+03)

    sec_scc = sec_run[9].section_single_configuration_calculation[0]
    assert len(sec_scc.energy_contributions) == 2
    assert sec_scc.energy_contributions[1].value.magnitude == approx(-1.00925367e-14)

    sec_scc = sec_run[3].section_single_configuration_calculation[0]
    assert sec_scc.stress_contributions[0].value[1][2].magnitude == approx(-1.42111377e+07)

    sec_scc = sec_run[1].section_single_configuration_calculation[0]
    assert sec_scc.stress_total.value[1][1].magnitude == approx(1.49076266e+08)

    sec_scc = sec_run[6].section_single_configuration_calculation[0]
    assert sec_scc.forces_total.value[5][2].magnitude == approx(-3.47924808e-10)

    sec_scc = sec_run[5].section_single_configuration_calculation[0]
    assert sec_scc.thermodynamics[0].pressure.magnitude == approx(2.52108927e+07)

    sec_scc = sec_run[2].section_single_configuration_calculation[0]
    assert sec_scc.x_fhi_vibes_pressure_kinetic.magnitude == approx(2.08283962e+08)

    sec_scc = sec_run[8].section_single_configuration_calculation[0]
    assert sec_scc.x_fhi_vibes_energy_potential_harmonic.magnitude == approx(4.08242214e-20)


def test_relaxation(parser):
    archive = EntryArchive()
    parser.parse('tests/data/relaxation.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'geometry_optimization'

    assert len(archive.section_run) == 1

    sec_attrs = archive.section_run[0].section_method[0].x_fhi_vibes_section_attributes[0]
    assert sec_attrs.x_fhi_vibes_attributes_timestep.magnitude == approx(1e-15)
    assert len(sec_attrs.x_fhi_vibes_section_attributes_atoms) == 1
    sec_atoms = sec_attrs.x_fhi_vibes_section_attributes_atoms[0]
    assert len(sec_atoms.x_fhi_vibes_atoms_symbols) == 2
    assert sec_atoms.x_fhi_vibes_atoms_masses.magnitude == approx(4.66362397e-26)

    sec_metadata = sec_attrs.x_fhi_vibes_section_attributes_metadata[0]
    sec_relaxation = sec_metadata.x_fhi_vibes_section_metadata_relaxation[0]
    assert sec_relaxation.x_fhi_vibes_relaxation_maxstep == 0.2
    assert not sec_relaxation.x_fhi_vibes_relaxation_hydrostatic_strain
    assert sec_relaxation.x_fhi_vibes_relaxation_type == 'optimization'

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 3
    assert sec_sccs[2].thermodynamics[0].volume.magnitude == approx(3.97721030e-29)
    assert sec_sccs[0].energy_contributions[1].value.magnitude == approx(-2.52313962e-15)


def test_molecular_dynamics(parser):
    archive = EntryArchive()
    parser.parse('tests/data/molecular_dynamics.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'molecular_dynamics'

    sec_attrs = archive.section_run[0].section_method[0].x_fhi_vibes_section_attributes[0]
    sec_md = sec_attrs.x_fhi_vibes_section_attributes_metadata[0].x_fhi_vibes_section_metadata_MD[0]
    assert sec_md.x_fhi_vibes_MD_md_type == 'Langevin'
    assert sec_md.x_fhi_vibes_MD_friction == 0.02

    sec_systems = archive.section_run[0].section_system
    assert len(sec_systems) == 11
    assert sec_systems[3].atom_positions[6][1].magnitude == approx(1.39537854e-10)
    assert sec_systems[7].atom_velocities[1][0].magnitude == approx(-249.97586102)
    assert sec_systems[2].lattice_vectors[0][2].magnitude == approx(2.20004000e-21)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert sec_sccs[4].x_fhi_vibes_heat_flux_0_harmonic[1].magnitude == approx(1.40863863e+13)
    assert sec_sccs[5].x_fhi_vibes_atom_forces_harmonic[3][0].magnitude == approx(8.40976902e-10)
    assert sec_sccs[6].x_fhi_vibes_momenta[7][2].magnitude == approx(-1.18929315e-24)


def test_phonon(parser):
    archive = EntryArchive()
    parser.parse('tests/data/phonopy.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'phonon'

    sec_attrs = archive.section_run[0].section_method[0].x_fhi_vibes_section_attributes[0]
    sec_phonon = sec_attrs.x_fhi_vibes_section_attributes_metadata[0].x_fhi_vibes_section_metadata_phonopy[0]
    assert sec_phonon.x_fhi_vibes_phonopy_version == '2.6.1'
    sec_atoms = sec_phonon.x_fhi_vibes_section_phonopy_primitive[0]
    assert np.shape(sec_atoms.x_fhi_vibes_atoms_positions) == (2, 3)
    assert sec_atoms.x_fhi_vibes_atoms_cell[0][2].magnitude == approx(2.70925272e-10)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 1
    assert sec_sccs[0].forces_total.value[6][1].magnitude == approx(-3.96793297e-11)
    assert sec_sccs[0].x_fhi_vibes_displacements[2][1].magnitude == approx(0.0)

    sec_system = archive.section_run[0].section_system[0]
    assert sec_system.atom_positions[3][2].magnitude == approx(5.41850544e-10)
    assert sec_system.lattice_vectors[1][1].magnitude == approx(5.41850544e-10)
