import warnings
from collections import UserDict
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace

from brainglobe_atlasapi.descriptors import (
    ANNOTATION_FILENAME,
    HEMISPHERES_FILENAME,
    MESHES_DIRNAME,
    METADATA_FILENAME,
    REFERENCE_FILENAME,
    STRUCTURES_FILENAME,
)
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_atlasapi.utils import read_json, read_tiff


class Atlas:
    """Base class to handle atlases in BrainGlobe.

    Parameters
    ----------
    path : str or Path object
        Path to folder containing data info.
    """

    left_hemisphere_value = 1
    right_hemisphere_value = 2

    def __init__(self, path):
        self.root_dir = Path(path)
        self.metadata = read_json(self.root_dir / METADATA_FILENAME)

        # Load structures list:
        structures_list = read_json(self.root_dir / STRUCTURES_FILENAME)
        # keep to generate tree and dataframe views when necessary
        self.structures_list = structures_list

        # Add entry for file paths:
        for struct in structures_list:
            struct["mesh_filename"] = (
                self.root_dir / MESHES_DIRNAME / "{}.obj".format(struct["id"])
            )

        self.structures = StructuresDict(structures_list)

        # Instantiate SpaceConvention object describing the current atlas:
        self.space = AnatomicalSpace(
            origin=self.orientation,
            shape=self.shape,
            resolution=self.resolution,
        )

        self._reference = None

        try:
            self.additional_references = AdditionalRefDict(
                references_list=self.metadata["additional_references"],
                data_path=self.root_dir,
            )
        except KeyError:
            warnings.warn(
                "This atlas seems to be outdated as no "
                "additional_references list "
                "is found in metadata!"
            )

        self._annotation = None
        self._hemispheres = None
        self._lookup = None

    @property
    def resolution(self):
        """Make resolution more accessible from class."""
        return tuple(self.metadata["resolution"])

    @property
    def orientation(self):
        """Make orientation more accessible from class."""
        return self.metadata["orientation"]

    @property
    def shape(self):
        """Make shape more accessible from class."""
        return tuple(self.metadata["shape"])

    @property
    def shape_um(self):
        """Make shape more accessible from class."""
        return tuple([s * r for s, r in zip(self.shape, self.resolution)])

    @property
    def hierarchy(self):
        """Returns a Treelib.tree object with structures hierarchy."""
        return self.structures.tree

    @property
    def lookup_df(self):
        """Returns a dataframe with id, acronym and name for each structure."""
        if self._lookup is None:
            self._lookup = pd.DataFrame(
                dict(
                    acronym=[r["acronym"] for r in self.structures_list],
                    id=[r["id"] for r in self.structures_list],
                    name=[r["name"] for r in self.structures_list],
                )
            )
        return self._lookup

    @property
    def reference(self):
        if self._reference is None:
            self._reference = read_tiff(self.root_dir / REFERENCE_FILENAME)
        return self._reference

    @property
    def annotation(self):
        if self._annotation is None:
            self._annotation = read_tiff(self.root_dir / ANNOTATION_FILENAME)
        return self._annotation

    @property
    def hemispheres(self):
        if self._hemispheres is None:
            # If reference is symmetric generate hemispheres block:
            if self.metadata["symmetric"]:
                # initialize empty stack:
                stack = np.full(self.metadata["shape"], 2, dtype=np.uint8)

                # Use bgspace description to fill out with hemisphere values:
                front_ax_idx = self.space.axes_order.index("frontal")

                # Fill out with 2s the right hemisphere:
                slices = [slice(None) for _ in range(3)]
                slices[front_ax_idx] = slice(
                    stack.shape[front_ax_idx] // 2 + 1, None
                )
                stack[tuple(slices)] = 1

                self._hemispheres = stack
            else:
                self._hemispheres = read_tiff(
                    self.root_dir / HEMISPHERES_FILENAME
                )
        return self._hemispheres

    def hemisphere_from_coords(self, coords, microns=False, as_string=False):
        """Get the hemisphere from a coordinate triplet.

        Parameters
        ----------
        coords : tuple or list or numpy array
            Triplet of coordinates. Default in voxels, can be microns if
            microns=True
        microns : bool
            If true, coordinates are interpreted in microns.
        as_string : bool
            If true, returns "left" or "right".


        Returns
        -------
        int or string
            Hemisphere label.

        """

        hem = self.hemispheres[self._idx_from_coords(coords, microns)]
        if as_string:
            hem = ["left", "right"][hem - 1]
        return hem

    def structure_from_coords(
        self,
        coords,
        microns=False,
        as_acronym=False,
        hierarchy_lev=None,
        key_error_string="Outside atlas",
    ):
        """Get the structure from a coordinate triplet.

        Parameters
        ----------
        coords : tuple or list or numpy array
            Triplet of coordinates.
        microns : bool
            If true, coordinates are interpreted in microns.
        as_acronym : bool
            If true, the region acronym is returned.
            If outside atlas (structure gives key error),
            return "Outside atlas"
        hierarchy_lev : int or None
            If specified, return parent node at thi hierarchy level.

        Returns
        -------
        int or string
            Structure containing the coordinates.
        """

        rid = self.annotation[self._idx_from_coords(coords, microns)]

        # If we want to cut the result at some high level of the hierarchy:
        if hierarchy_lev is not None:
            rid = self.structures[rid]["structure_id_path"][hierarchy_lev]

        if as_acronym:
            try:
                d = self.structures[rid]
                return d["acronym"]
            except KeyError:
                return key_error_string
        else:
            return rid

    def structures_from_coords_list(
        self,
        coords,
        microns=False,
        as_acronym=False,
        key_error_string="Outside atlas",
    ):
        """Get the list of structures from a list of coordinate triplets.

        Parameters
        ----------
        coords : np.ndarray
            List of triplets of coordinates.
        microns : bool, optional
            If true, coordinates are interpreted in microns. Default is False.
        as_acronym : bool, optional
            If true, the region acronym is returned.
            If outside atlas (structure gives key error),
            return "Outside atlas". Default is False.
        key_error_string : str, optional
            Acronym to return if coord is outside the brain. Default is
            "Outside atlas".

        Returns
        -------
        int or string
            Structure containing the coordinates.
        """
        # convert input to a numpy array
        if not isinstance(coords, np.ndarray):
            try:
                coords = np.array(coords)
            except ValueError:
                raise ValueError("some element in coords are not triplets.")

        # checks
        if coords.ndim < 2:
            # this might be only one triplet
            if coords.shape[0] == 3:
                warnings.warn(
                    "coords interpreted as a single triplet of coordinates"
                )
                # use regular method for single triplet
                return self.structure_from_coords(
                    coords,
                    microns,
                    as_acronym=as_acronym,
                    key_error_string=key_error_string,
                )
            else:
                raise ValueError("coords is not a triplet of coordinates")
        elif coords.ndim == 2:
            if coords.shape[1] != 3:
                # elements in the input list do not contain 3 coordinates
                raise ValueError(
                    "coords is not an array of triplets of coordinates"
                )
        else:
            raise ValueError(
                "coords is not an array of triplets of coordinates"
            )

        # get atlas lims
        if microns:
            lims = self.shape_um
        else:
            lims = self.shape

        # reformat so we end up with :
        # coords[0] -> x0, x1, x2, ... cords[1] -> y0, y1, ...
        coords = coords.T
        coords = [coords[0], coords[1], coords[2]]

        # set out-of-atlas objects at 0
        for idaxis in range(0, len(lims)):
            coords[idaxis][
                (coords[idaxis] >= lims[idaxis]) | (coords[idaxis] < 0)
            ] = 0

        # convert to pixel indices if coords is in microns
        if microns:
            coords = [
                (np.array(coords[idx]) / self.resolution[idx]).astype(int)
                for idx in range(0, len(self.resolution))
            ]

        # convert i, j, k indices in raveled indices
        linear_indices = np.ravel_multi_index(
            coords, dims=self.annotation.shape
        )
        # get the structure id from the annotation stack
        idlist = self.annotation.ravel()[linear_indices]

        # get corresponding acronyms
        if as_acronym:
            lookup = self.lookup_df.set_index("id")
            # add index 0
            lookup.loc[0] = [key_error_string, key_error_string]
            return lookup.loc[idlist, "acronym"].values.tolist()
        else:
            return idlist.tolist()

    # Meshes-related methods:
    def _get_from_structure(self, structure, key):
        """Internal interface to the structure dict. It support querying with a
        single structure id or a list of ids.

        Parameters
        ----------
        structure : int or str or list
            Valid id or acronym, or list if ids or acronyms.
        key : str
            Key for the Structure dictionary (eg "name" or "rgb_triplet").

        Returns
        -------
        value or list of values
            If structure is a list, returns list.

        """
        if isinstance(structure, list) or isinstance(structure, tuple):
            return [self._get_from_structure(s, key) for s in structure]
        else:
            return self.structures[structure][key]

    def mesh_from_structure(self, structure):
        return self._get_from_structure(structure, "mesh")

    def meshfile_from_structure(self, structure):
        return self._get_from_structure(structure, "mesh_filename")

    def root_mesh(self):
        return self.mesh_from_structure("root")

    def root_meshfile(self):
        return self.meshfile_from_structure("root")

    def _idx_from_coords(self, coords, microns):
        # If microns are passed, convert:
        if microns:
            coords = [c / res for c, res in zip(coords, self.resolution)]

        return tuple([int(c) for c in coords])

    def get_structure_ancestors(self, structure):
        """Returns a list of acronyms for all ancestors of a given structure

        Parameters
        ----------
        structure : str or int
            Structure id or acronym

        Returns
        -------
        list
            List of descendants acronyms

        """
        ancestors_id = self._get_from_structure(
            structure, "structure_id_path"
        )[:-1]

        return self._get_from_structure(ancestors_id, "acronym")

    def get_structure_descendants(self, structure):
        """Returns a list of acronyms for all descendants of a given structure.

        Parameters
        ----------
        structure : str or int
            Structure id or acronym

        Returns
        -------
        list
            List of descendants acronyms

        """
        structure = self._get_from_structure(structure, "acronym")

        # For all structures check if given structure is ancestor
        descendants = []
        for struc in self.structures.keys():
            if structure in self.get_structure_ancestors(struc):
                descendants.append(self._get_from_structure(struc, "acronym"))

        return descendants

    def get_structure_mask(self, structure):
        """
        Returns a stack with the mask for a specific structure (including all
        sub-structures).

        This function might take a few seconds for structures with many
        children.

        Parameters
        ----------
        structure : str or int
            Structure id or acronym

        Returns
        -------
        np.array
            stack containing the mask array.
        """
        structure_id = self.structures[structure]["id"]
        descendants = self.get_structure_descendants(structure)

        descendant_ids = [
            self.structures[descendant]["id"] for descendant in descendants
        ]
        descendant_ids.append(structure_id)

        mask_stack = np.zeros(self.shape, self.annotation.dtype)
        mask_stack[np.isin(self.annotation, descendant_ids)] = structure_id

        return mask_stack


class AdditionalRefDict(UserDict):
    """Class implementing the lazy loading of secondary references
    if the dictionary is queried for it.
    """

    def __init__(self, references_list, data_path, *args, **kwargs):
        self.data_path = data_path
        self.references_list = references_list

        super().__init__(*args, **kwargs)

    def __getitem__(self, ref_name):
        if ref_name not in self.keys():
            if ref_name not in self.references_list:
                warnings.warn(
                    f"No reference named {ref_name} "
                    f"(available: {self.references_list})"
                )
                return None

            self.data[ref_name] = read_tiff(
                self.data_path / f"{ref_name}.tiff"
            )

        return self.data[ref_name]
