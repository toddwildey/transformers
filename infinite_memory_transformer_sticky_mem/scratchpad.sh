# Build "gpt2_infinity_config.shared.json" from keys missing in "gpt2_config.json"
cat gpt2_infinity_config.json \
    | ./diff_model_config_json.sh "gpt2_config.json" \
    | tee gpt2_infinity_config.shared.json

# Override "gpt2_config.json" with "gpt2_infinity_config.shared.json"
./override_model_config_json.sh gpt2_config.json gpt2_infinity_config.shared.json \
    | tee gpt2_infinity_config.json

# Override "gpt2-large_config.json" with "gpt2_infinity_config.shared.json"
./override_model_config_json.sh gpt2-large_config.json gpt2_infinity_config.shared.json \
    | tee gpt2-large_infinity_config.json

# Override "gpt2-xl_config.json" with "gpt2_infinity_config.shared.json"
./override_model_config_json.sh gpt2-xl_config.json gpt2_infinity_config.shared.json \
    | tee gpt2-xl_infinity_config.json
