NIST SPECIAL DATABASE 302h
==========================

ABOUT
-----
NIST Special Database (SD) 302 is a collection of biometric data collected at
the Intelligence Advanced Research Project Activity's Nail to Nail (N2N)
Fingerprint Challenge. SD 302h specifically consists of Electronic Biometric
Transmission Specification (EBTS) records of latent distal phalanx images with
Extended Feature Set (EFS) Profile 2 annotations and ridge quality maps,
formatted as Latent Friction Ridge Features Search (LFFS) transactions. For
complete details, please refer to NIST Technical Note 2190 [1] and
NIST Technical Note 2367 [2]. For details about other portions of SD 302, please
refer to  NIST Technical Note 2007 [3]. For details about the N2N Fingerprint
Challenge itself, please refer to NIST Interagency Report 8210 [4].

ERRATA
------
For a growing list of known errata in this version of the dataset, please visit:
https://www.nist.gov/document/erratasd302txt

DIRECTORY STRUCTURE
-------------------
The directory structure of SD 302h is as follows:

ebts                                              # Record format
├── checksum_latent_lffs_enhanced_masked.csv      # File checksums
├── checksum_latent_lffs_enhanced_unmasked.csv
├── checksum_latent_lffs_original_masked.csv
├── checksum_latent_lffs_original_unmasked.csv
└── latent                                        # Collection type
    └── lffs                                      # Transaction type
        ├── enhanced                              # Image color enhancements
        │   ├── masked                            # Pixels masking outside ROI
        │   │   └── 1000                          # Resolution, in PPI
        │   │       ├── 00002302                  # Study participant identifier
        │   │       │   ├── 00002302_1A_R_L01_BP_S04_1000PPI_8BPC_1CH_
        │   │       │   │       LP01-1_1.lffs
        │   │       │   ├── 00002302_1D_L_L01_BP_S04_1000PPI_8BPC_1CH_
        │   │       │   │       LP01-1_1.lffs
        │   │       │   ├── 00002302_1D_L_L01_BP_S04_1000PPI_8BPC_1CH_
        │   │       │   │       LP02-1_1.lffs
        │   │       │   └── ...
        │   │       └── ...
        │   └── unmasked
        └── original
            └── ...

This directory structure was chosen to allow for NIST to easily deliver future
versions of the same images in different file formats, as well as images from
other modalities and device types alongside this distribution.

The topmost `ebts` directory contains a directory for each of the collection
types, from the N2N Fingerprint Challenge, limited to `latent` in SD 302h. The
directories under the collection type are named after the type of transaction
encoded in the EBTS records. Under each type of transaction are directories of
the same image with different types of image enhancement techniques applied,
limited to `original` (i.e., unprocessed) and `enhanced` (i.e., enhancements to
visually boost ridge structure applied). Under each, there are `unmasked`
versions that do not fill areas outside of the region of interest with white
pixels, as well as `masked` versions that do. Records are then separated by the
resolution (in ppi) of the latent image and the study participant identifier.

Image names are in the form SUBJECT_ACTIVITY_HAND_ENCOUNTER_TECHNIQUE_DIGITIZER_
RESOLUTION_DEPTH_CHANNELS_LPNUMBER-IMP_SOURCE.EXT, where:

   SUBJECT: Unique identifier for this study participant.
  ACTIVITY: Activity performed to leave this latent impression. For a complete
            list of activities and their descriptions, refer to NIST Interagency
            Report 8210, Section 5.1 [4].
      HAND: L for left hand, R for right hand, or X if unknown.
 ENCOUNTER: A unique number to represent a particular encounter that was
            developed from this study participant and ACTIVITY.
 TECHNIQUE: The technique used to expose the print in this image. For a complete
            list of techniques and their descriptions, refer to NIST Interagency
            Report 8210, Section 5.2 [4]. This field is abbreviated, with BP
            meaning black powder, IN meaning 1,2-Indanedione, WT and BT
            meaning adhesive-side powder (white and black, respectively), and CA
            meaning cyanoacrylate.
 DIGITIZER: The device used to digitize this image. For a complete list of
            devices and their descriptions, refer to NIST Interagency Report
            8210, Section 5.3 [4]. Multiple flatbed scanners were used,
            indicated by S#. Only one piece of hardware was used for other
            digitization methods.
RESOLUTION: The capture resolution of the image, in pixels per inch.
     DEPTH: The number of bits in a single color channel.
  CHANNELS: The number of color channels represented in a single pixel. 1
            indicates grayscale and 3 represents color in a red, green, and
            blue arrangement.
    NUMBER: An identifier to represent an individual latent print of value from
            this ENCOUNTER.
       IMP: An identifier to indicate the impression within the image. For most
            records, this is 1, indicating a single impression is visible.
    SOURCE: The likely source of the latent print, with 1 for distal phalanx,
            2 for other phalanx, 3 for palm, and 4 being unknown.
       EXT: File format extension.

Comma-separated value (CSV) files accompany every directory of images. Contained
in this file are the SHA 256 checksums of the files contained within the named
directory.

ACKNOWLEDGMENTS
---------------
The authors would like to thank the NIST Special Programs Office's Forensic
Science Research Program for funding this activity, as well as the Intelligence
Advanced Research Projects Activity for funding the initial collection of data
in SD 302.

LICENSE
-------
Users of SD 302 shall adhere to all terms agreed to upon obtaining SD 302.

DISCLAIMER
----------
Certain commercial equipment, instruments, or materials are identified in this
document in order to specify the development procedure adequately. Such
identification is not intended to imply recommendation or endorsement by the
National Institute of Standards and Technology, nor is it intended to imply
that the materials or equipment identified are necessarily the best available
for the purpose.

[1] Fiumara G, Schwarz M, Heising J, Peterson J, Flanagan P, Marshall K (2021)
    NIST Special Database 302: Supplemental Release of Latent Annotations. NIST
    Technical Note 2190. https://doi.org/10.6028/NIST.TN.2190

[2] Fiumara G, Schwarz M, Heising J, Peterson J, Ko K, Flanagan P, Marshall K
    (2026) NIST Special Database 302: Annotated Latent Distal Phalanges. NIST
    Technical Note 2367. https://doi.org/10.6028/NIST.TN.2367

[3] Fiumara G, Flanagan P, Grantham J, Ko K, Marshall K, Schwarz M, Tabassi E,
    Woodgate B, Boehnen C (2019) NIST Special Database 302: Nail to Nail
    Fingerprint Challenge. NIST Technical Note 2007.
    https://doi.org/10.6028/NIST.TN.2007

[4] Fiumara G, Tabassi E, Flanagan P, Grantham J, Ko K, Marshall K, Schwarz M,
    Woodgate B, Boehnen C (2018) Nail to Nail Fingerprint Challenge: Prize
    Analysis. NIST Interagency Report 8210. https://doi.org/10.6028/NIST.IR.8210
