# Build "gpt2_iterator_config.shared.json" from keys missing in "gpt2_config.json"
cat gpt2_iterator/gpt2-large_iterator_config.json \
    | ./gpt2_iterator/diff_model_config_json.sh "gpt2_iterator/gpt2-large_config.json" \
    | tee gpt2_iterator/gpt2_iterator_config.shared.json

# Override "gpt2_config.json" with "gpt2_iterator_config.shared.json"
./gpt2_iterator/override_model_config_json.sh \
        gpt2_iterator/gpt2_config.json \
        gpt2_iterator/gpt2_iterator_config.shared.json \
    | tee gpt2_iterator/gpt2_iterator_config.json

# Override "gpt2-large_config.json" with "gpt2_iterator_config.shared.json"
./gpt2_iterator/override_model_config_json.sh \
        gpt2_iterator/gpt2-large_config.json \
        gpt2_iterator/gpt2_iterator_config.shared.json \
    | tee gpt2_iterator/gpt2-large_iterator_config.json

# Override "gpt2-xl_config.json" with "gpt2_iterator_config.shared.json"
./gpt2_iterator/override_model_config_json.sh \
        gpt2_iterator/gpt2-xl_config.json \
        gpt2_iterator/gpt2_iterator_config.shared.json \
    | tee gpt2_iterator/gpt2-xl_iterator_config.json
