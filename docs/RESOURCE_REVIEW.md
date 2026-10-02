# Initial resource and focused prior-work review

Checked 2026-10-02 against primary sources. This is a focused starting review, not a systematic novelty review or an approval to use/redistribute data. No datasets were downloaded or access forms submitted.

## BrainWeb anatomical models

The [20 normal anatomical models](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html) provide 3D fuzzy tissue fractions and dominant-tissue labels. They supply structure, not measured ADC. Assign diffusion parameters explicitly as simulation assumptions.

The anatomy page, [home page](https://brainweb.bic.mni.mcgill.ca/brainweb/), [FAQ](https://brainweb.bic.mni.mcgill.ca/brainweb/faq.html), and an official example download form did not expose an explicit redistribution grant in this review. Downloadability must not be described as a verified permissive data license. Keep source and anatomically derived files outside Git until terms are clarified, and retain the requested source citations. Use a generated geometric phantom for initial software checks.

## DIPY simulation tools

The [simulation API](https://docs.dipy.org/stable/reference/dipy.sims.html#dipy.sims.voxel.add_noise) and [official implementation](https://docs.dipy.org/stable/_modules/dipy/sims/voxel.html#add_noise) support Gaussian, Rician, and Rayleigh noise with a supplied random generator. The Rician path uses independent Gaussian real/imaginary components, `sigma=S0/SNR`, and magnitude formation. It returns magnitude samples rather than a preserved complex or multicoil k-space acquisition. The initial project approximation is compatible with this construction; coil-aware extensions require explicit modeling.

The [DIPY license](https://github.com/dipy/dipy/blob/master/LICENSE) permits software redistribution/modification under BSD-style conditions, including retention of notices and no endorsement; check file-specific exceptions. Pin the actual release before use rather than relying on changing `stable`/`master` links. Software licensing does not license external datasets. DIPY is not installed as a project dependency yet.

## fastMRI Prostate: paper and access boundary

The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11032332/) reports 312 patients and a suggested 218/48/46 split. DWI uses axial 2D EPI, with four b50 and twelve b1000 repetitions for each of three directions. Raw k-space includes a 50-entry diffusion axis: twelve b50, thirty-six b1000, and two b0 measurements, with separate slice/coil/readout/phase axes. The release includes reconstruction/calibration information. Verify each selected file's metadata before encoding its acquisition contract.

The [official access portal](https://fastmri.med.nyu.edu/) requires an application and agreement acceptance and indicates dataset-specific agreements. No user approval or prostate-specific executed agreement has been verified. Review the exact terms presented during the prostate application before Stage 2. Keep patient data, private derived maps, and download links out of public artifacts; do not presume image-publication permission. The [reconstruction repository's software license](https://github.com/cai2r/fastMRI_prostate) does not confer dataset rights.

**Project inference:** reserve reference repetitions before defining input budgets. A reference reconstructed from all repetitions overlaps reduced-budget inputs and is not independent. The patient reference remains noisy; absolute accuracy claims require simulation truth under explicit assumptions.

## Pfaff et al. and candidate contribution

[Pfaff et al., fastMRI Prostate denoising/ADC study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11484701/) already compares SURE, Noise2Noise, Half2Full, and MP-PCA and tests reduced inputs of two b50/six b1000 repetitions per direction. It examines repetition counts and low-field transfer, with quantitative residual-based evaluation. Acquisition reduction and ADC denoising alone are therefore insufficient novelty claims.

The reviewed full text did not report known-truth focal ADC-change recovery, calibrated ADC-error bounds, or an adaptive acquisition stopping policy. SURE's signal-MSE estimation does not establish calibrated patient-specific ADC uncertainty. This limited observation is not proof of absence from the wider literature.

**Candidate contribution, pending wider review:** benchmark focal quantitative change preservation and falsely confident acquisition stopping across known SNR, geometry, contrast, and budget regimes. Before choosing the final model/publication claim, extend the search to quantitative denoising bias, small-feature suppression, calibrated diffusion estimation, and sequential/adaptive acquisition. Record inclusion criteria, search dates, exact papers, code availability, and overlaps.
