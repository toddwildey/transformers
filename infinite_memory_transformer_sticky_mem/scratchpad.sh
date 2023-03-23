# Build "gpt2_infinity_base_config.json" from keys missing in "gpt2_config.json"
cat gpt2_infinity_config.json \
    | ./diff_model_config_json.sh "gpt2_config.json" \
    | tee gpt2_infinity_base_config.json

# Override "gpt2_config.json" with "gpt2_infinity_base_config.json"
./override_model_config_json.sh gpt2_config.json gpt2_infinity_base_config.json \
    | tee gpt2_infinity_config.json
