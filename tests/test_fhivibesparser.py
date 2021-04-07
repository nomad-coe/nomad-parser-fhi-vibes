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

    assert pytest.approx(sec_run[8].section_system[0].atom_positions[3][2].magnitude, 5.30098546e-10)
    assert pytest.approx(sec_run[5].section_system[0].atom_velocities[1][0].magnitude, -2.18864066e+03)

    sec_scc = sec_run[9].section_single_configuration_calculation[0]
    assert len(sec_scc.section_energy_contribution) == 2
    assert pytest.approx(sec_scc.section_energy_contribution[1].energy_contribution_value.magnitude, -1.00925367e-14)

    sec_scc = sec_run[3].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.section_stress_tensor_contribution[0].stress_tensor_contribution_value[1][2].magnitude, -1.42111377e+07)

    sec_scc = sec_run[1].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.stress_tensor[1][1].magnitude, 1.49076266e+08)

    sec_scc = sec_run[6].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.atom_forces[5][2].magnitude, -3.47924808e-10)

    sec_scc = sec_run[5].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.pressure.magnitude, 2.52108927e+07)

    sec_scc = sec_run[2].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.x_fhi_vibes_pressure_kinetic.magnitude, 2.08283962e+08)

    sec_scc = sec_run[8].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.x_fhi_vibes_energy_potential_harmonic.magnitude, 4.08242214e-20)


def test_relaxation(parser):
    archive = EntryArchive()
    parser.parse('tests/data/relaxation.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'geometry_optimization'

    assert len(archive.section_run) == 1

    sec_attrs = archive.section_run[0].section_method[0].x_fhi_vibes_section_attributes[0]
    assert pytest.approx(sec_attrs.x_fhi_vibes_attributes_timestep.magnitude, 1e-15)
    assert len(sec_attrs.x_fhi_vibes_section_attributes_atoms) == 1
    sec_atoms = sec_attrs.x_fhi_vibes_section_attributes_atoms[0]
    assert len(sec_atoms.x_fhi_vibes_atoms_symbols) == 2
    assert pytest.approx(sec_atoms.x_fhi_vibes_atoms_masses.magnitude, 4.66362397e-26)

    sec_metadata = sec_attrs.x_fhi_vibes_section_attributes_metadata[0]
    sec_relaxation = sec_metadata.x_fhi_vibes_section_metadata_relaxation[0]
    assert sec_relaxation.x_fhi_vibes_relaxation_maxstep == 0.2
    assert not sec_relaxation.x_fhi_vibes_relaxation_hydrostatic_strain
    assert sec_relaxation.x_fhi_vibes_relaxation_type == 'optimization'

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 3
    assert pytest.approx(sec_sccs[2].x_fhi_vibes_volume.magnitude, 3.97721030e-29)
    assert pytest.approx(sec_sccs[0].section_energy_contribution[0].energy_contribution_value.magnitude, -2.52313962e-15)


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
    assert pytest.approx(sec_systems[3].atom_positions[6][1].magnitude, 1.39537854e-10)
    assert pytest.approx(sec_systems[7].atom_velocities[1][0].magnitude, -249.97586102)
    assert pytest.approx(sec_systems[2].lattice_vectors[0][2].magnitude, 2.20004000e-21)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert pytest.approx(sec_sccs[4].x_fhi_vibes_heat_flux_0_harmonic[1].magnitude, 1.40863863e+13)
    assert pytest.approx(sec_sccs[5].x_fhi_vibes_atom_forces_harmonic[3][0].magnitude, 8.40976902e-10)
    assert pytest.approx(sec_sccs[6].x_fhi_vibes_momenta[7][2].magnitude, -1.23261549e-22)


def test_phonon(parser):
    archive = EntryArchive()
    parser.parse('tests/data/phonopy.nc', archive, None)

    assert archive.section_workflow.workflow_type == 'phonon'

    sec_attrs = archive.section_run[0].section_method[0].x_fhi_vibes_section_attributes[0]
    sec_phonon = sec_attrs.x_fhi_vibes_section_attributes_metadata[0].x_fhi_vibes_section_metadata_phonopy[0]
    assert sec_phonon.x_fhi_vibes_phonopy_version == '2.6.1'
    sec_atoms = sec_phonon.x_fhi_vibes_section_phonopy_primitive[0]
    assert np.shape(sec_atoms.x_fhi_vibes_atoms_positions) == (2, 3)
    assert pytest.approx(sec_atoms.x_fhi_vibes_atoms_cell[0][2].magnitude, 2.70925272e-10)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 1
    assert pytest.approx(sec_sccs[0].atom_forces[6][1].magnitude, -3.96793297e-11)
    assert pytest.approx(sec_sccs[0].x_fhi_vibes_displacements[2][1].magnitude, 0.0)

    sec_system = archive.section_run[0].section_system[0]
    assert pytest.approx(sec_system.atom_positions[3][2].magnitude, 5.41850544e-10)
    assert pytest.approx(sec_system.lattice_vectors[1][1].magnitude, 5.41850544e-10)
