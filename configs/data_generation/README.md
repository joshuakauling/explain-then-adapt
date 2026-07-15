# Data-Generation Configurations

The rebuilt data-generation pipeline currently uses explicit command-line
arguments and provider-independent request JSONL files. This avoids inventing
model names, paths, or sampling defaults that cannot yet be tied to a specific
historical run.

Once the final run manifests have been reconstructed, versioned configuration
files in this directory should contain:

- prompt versions and model identifiers,
- sampling settings for initial generation, judging, and rewriting,
- few-shot pool and selection settings,
- the accepted augmentation target,
- references to dataset manifests by version or checksum.

API keys, local model paths, batch job identifiers, generated requests, provider
responses, and accepted datasets remain local artifacts outside Git.
