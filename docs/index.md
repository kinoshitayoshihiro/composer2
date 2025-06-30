# Composer2 Documentation

Welcome to the documentation for **Composer2**, a framework that merges poetic Japanese narration with emotive music generation. This project automatically composes instrumental parts and rhythmic backing for each chapter of your story.

Browse the API reference to learn how to integrate the generators into your workflow.

See [Groove Sampler](groove_sampler.md) for training drum models.
Auxiliary conditioning and deterministic sampling are covered in
[Aux Features](aux_features.md).
For an example of emotion-driven bass parts see the *Bass Generator Usage*
section in the project README.

Install optional extras for the GUI:
`pip install -e .[gui]`.

For a live comparison of the n-gram and RNN models check out the
Streamlit GUI (run with `modcompose gui`):


Real-time playback is available via `modcompose realtime`.
