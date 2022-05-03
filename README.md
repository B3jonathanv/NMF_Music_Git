# NMF for Music

This repository accompanies Jonathan Valyou's honors thesis entitled "Nonnegative Matrix Factorization for Music - Tuning the NMF Algorithm with
Regularization" which can be located in the Emory University Honors Thesis Archives at the following link: https://etd.library.emory.edu/concern/etds/np193b40x?locale=en.  Much of this code was adapted from the NMF Toolbox repository: https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/#Python.  This repository explores how Nonnegative Matrix Factorization(NMF) can be utilized in source separation in music and introduces Regularized NMF algorithm code to aid in denoising audio and promoting sparsity.

## Installation

```python

git clone https://github.com/B3jonathanv/NMF_Music_Git.git

```

## Citation

If this repository is utilized and/or any information contained within the thesis entitled "Nonnegative Matrix Factorization for Music - Tuning the NMF Algorithm with Regularization", please include the following citation:

Valyou Jonathan.  Nonnegative Matrix Factorization for Music - Tuning the NMF Algorithm with
Regularization.  Emory Theses and Dissertations Repository.  3 May 2022.



## Examples

<img width="672" alt="Visualization_Guide copy" src="https://user-images.githubusercontent.com/72425355/162584465-e5e09a60-d6ca-41c2-baf2-b9564d02bd20.png">
The spectrogram X is the bottom right corner with axes of time(seconds) and frequency(Hz). The visual representation of the frequency data for each source W is in the bottom left of the diagram. The visual representation of the temporal data for each source H is in the top right corner of the diagram.


![C_Scale_No_Reg](https://user-images.githubusercontent.com/72425355/162584215-fb777153-b67f-4a95-bb1c-e040812b9f0f.png)
This is the source separation performed on the eight notes of the C Major Scale to demonstrate how NMF can separate note pitches on the same instrument.  The NMFD algorithm was utilized with input parameters of 8 sources for 8 distinct pitches, 300 iterations, 8 Template Frames, randomly initialized W, and uniformly initialized H.  Below is a convergence analysis of this source separation example.


![Convergence_ObjFunc_Scale](https://user-images.githubusercontent.com/72425355/162584238-c84cbe87-600a-4963-9639-98a514fd9843.png)
![Convergence_W_Scale](https://user-images.githubusercontent.com/72425355/162584247-a28512ee-aff0-4d71-80a0-adfab2b6e106.png)
![Convergence_H_Scale](https://user-images.githubusercontent.com/72425355/162584245-b6b65bf8-33c2-43bc-be93-49e12447f9f6.png)


![DrumKick](https://user-images.githubusercontent.com/72425355/162584268-758377b0-698b-4abc-9294-b5a96a6690bc.png)
This is source separation performed on an audio of three percussion instruments to demonstrate how NMF can separate instruments of distinct frequency ranges.  The NMFD algorithm was utilized with input parameters of 3 sources for 3 distinct instruments, 30 iterations, 8 Template Frames, randomly initialized W, and uniformly initialized H. The three colors represent each of the three percussion instruments: red represents the kick drum, green represents the snare drum, and blue represents the ride cymbal.


![Bach_Choral_NMF](https://user-images.githubusercontent.com/72425355/162584228-d91950c6-8abf-4d71-829c-8135ba86c6c7.png)
This is the source separation performed on a recording of Ein feste Burg ist unser Gott to demonstrate how NMF handles source separation for more complex,polyphonic musical arrangements.  The NMFD algorithm was utilized with input parameters of 8 sources for 8 distinct pitches, 200 iterations, 8 Template Frames, randomly initialized W, and uniformly initialized H.



![Coughing_NMF](https://user-images.githubusercontent.com/72425355/162584202-43360ec5-77a8-4c32-bdd5-affa7aaf9e62.png)
This is the source separation performed on a short recording of a person coughing over a sustained symphony note to demonstrate how NMF can separate out distinct non-uniform, non-Gaussian noise from an audio.  The NMF algorithm with no regularization parameter was utilized with input parameters of 2 sources for the music and the noise, 200 iterations, randomly initialized W, and uniformly initialized H. Blue corresponds with the coughing noise and red corresponds with the orchestra.


The below example demonstrates how Regularized NMF can aid source separation of an audio file that is perturbed by random Gaussian noise.
```python

run Reg_Script.py

```
![No_Reg_Noise_Scale](https://user-images.githubusercontent.com/72425355/162584134-0392704f-1a03-4b10-a936-16b36a4e98e5.png)
The NMF algorithm with no regularization parameter was utilized with input parameters of 8 sources for 8 distinct pitches, 50 iterations, randomly initialized W, and uniformly initialized H.

![1H_Scale_Reg](https://user-images.githubusercontent.com/72425355/162584147-295fdd60-dab1-4826-84e0-fe192fb8af6f.png)
The Regularized NMF algorithm with regularization expression γ∥H∥_1 was utilized with input parameters of a regularization parameter of γ = 5×10^−6, 8 sources, 50 iterations, randomly initialized W, and uniformly initialized H.



## References

Patricio Lopez-Serrano, Christian Dittmar, Yigitcan Ozer, and Meinard Muller.
NMF Toolbox, 2019.

Patricio Lopez-Serrano, Christian Dittmar, Yigitcan Ozer, and Meinard Muller.
NMF Toolbox: Music processing applications of nonnegative matrix factorization.
In Proceedings of the International Conference on Digital Audio Effects (DAFx),
Birmingham, UK, September 2019.

Paris Smaragdis and J. Brown. Non-negative matrix factor deconvolution; ex-
traction of multiple sound sources from monophonic inputs. volume 3195, 09
2004.

Daniel D. Lee and H. Sebastian Seung. Learning the parts of objects by non-
negative matrix factorization. Nature, 401(6755):788–791, 1999.

Stanford University Department of Music. Sound examples.


