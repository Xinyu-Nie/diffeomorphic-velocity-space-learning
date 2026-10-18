# BrainWeb source attribution

The illustrative meshes in this directory were derived from a synthetic T1 image
from the [BrainWeb Simulated Brain Database](https://brainweb.bic.mni.mcgill.ca/brainweb/).
The supplied image identifier was `t1_icbm_normal_1mm_pn0_rf0.mnc.gz`
(normal ICBM phantom, 1 mm, 0% noise and 0% intensity nonuniformity).

FreeSurfer reconstruction supplied the left pial, inflated and registered spherical
surfaces. The pial and inflated surfaces were resampled to ico5 and ico6 and
processed for this illustrative matching example. These are reconstructed and
processed meshes, not BrainWeb's original anatomical ground-truth surfaces.
The exact FreeSurfer version and reconstruction command are not recorded with
the supplied reconstruction.

The repository's MIT license applies to its software. It does not grant rights
over the underlying BrainWeb data. Retain this attribution with the derived
assets and consult the BrainWeb site for the source data's terms and requested
citations. The original MRI and native FreeSurfer files are not included.

BrainWeb requests acknowledgment of the database and its methods, including:

- C. A. Cocosco, V. Kollokian, R. K.-S. Kwan, A. C. Evans.
  *BrainWeb: Online Interface to a 3D MRI Simulated Brain Database.*
  NeuroImage 5(4), part 2/4, S425, 1997.
- R. K.-S. Kwan, A. C. Evans, G. B. Pike.
  *MRI simulation-based evaluation of image-processing and classification methods.*
  IEEE Transactions on Medical Imaging 18(11), 1085–1097, 1999.
- D. L. Collins, A. P. Zijdenbos, V. Kollokian, J. G. Sled, N. J. Kabani,
  C. J. Holmes, A. C. Evans.
  *Design and Construction of a Realistic Digital Brain Phantom.*
  IEEE Transactions on Medical Imaging 17(3), 463–468, 1998.
