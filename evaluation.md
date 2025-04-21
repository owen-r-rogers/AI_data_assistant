# Evaluation

## Issues

The biggest issue I had here was getting the repo actually set up, I was getting an error with git push. This was resolved by downloading GitHub Desktop, which worked, however I did get a security breach alert that the .env file was made public (so I apologize for getting the API key reset again). The .env is in the .gitignore, so I'm not sure why it was added in the first place, and I was careful in resolving the issue with GitHub, so I'm still pretty confused on how it was uploaded as a commit.

Another issue I had was with the actual functions that were part of the pipeline - since this performs a BLAST run using a server hosted by the NIH, sometimes there was a lot of traffic so getting the BLAST run to be successfully was the rate limiting step.