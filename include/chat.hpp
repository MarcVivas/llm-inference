#pragma once
#include "model.hpp"
#include "model_inference.hpp"
#include <iostream>
#include <string>
#include <vector>


struct ChatMessage {
    std::string role;  // User or assistant
    std::string content;
};

class Chat {


    public:
        Chat(const Model& model, ModelInference& model_inference)
            :model(model), inference_engine(model_inference)
        {
            // Resolve special stop tokens dynamically from the tokenizer
            im_end_id_     = resolve_special_token("<|im_end|>");
            endoftext_id_  = resolve_special_token("<|endoftext|>");

            // Seed default system prompt
            append_to_history("system", "You are a helpful, concise assistant.");
        }

        // Interactive Terminal REPL Loop
        void start_interactive_session() {
            std::println("\n========================================================");
            std::println("      Chat Session Ready! Type 'exit' or 'quit' to end  ");
            std::println("========================================================\n");
        
            std::string user_input;
            while (true) {
                std::print("User: ");
                if (!std::getline(std::cin, user_input)) {
                    break;
                }
        
                if (user_input.empty()) {
                    continue;
                }
        
                if (user_input == "exit" || user_input == "quit") {
                    std::println("Exiting chat. Goodbye!");
                    break;
                }
        
                generate_response(user_input, model.config.max_seq_len);
                std::println("");
            }
        }


    private:
        ModelInference& inference_engine;
        const Model& model;
        std::vector<ChatMessage> history_;
        int32_t im_end_id_{-1};
        int32_t endoftext_id_{-1};

        int32_t resolve_special_token(const std::string& token_str) const {
            std::vector<int> ids = model.tokenizer->Encode(token_str);
            return ids.empty() ? -1 : ids[0];
        }

        bool is_stopping_token(int32_t token_id) const {
            return (token_id == im_end_id_) || (token_id == endoftext_id_);
        }
        
        void append_to_history(const std::string& role, const std::string& content) {
            history_.push_back(ChatMessage{.role = role, .content = content});
        }

        // Formats conversation history into the standard ChatML template:
        // <|im_start|>role\ncontent<|im_end|>\n
        std::string format_chat_template() const {
            std::string formatted_prompt;
        
            for (const auto& msg : history_) {
                formatted_prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
            }
        
            // Cue the model for the assistant's turn
            formatted_prompt += "<|im_start|>assistant\n";
            return formatted_prompt;
        }
        
        // Fromat the entire conversation into template string
        std::string build_prompt_with_template(){

        }

        // Check if a token signals the end of turn
        bool is_eos_token(int token_id) const{
            return false;
        }

        // Streams generation token-by-token
        std::string generate_response(const std::string& user_prompt, size_t max_new_tokens) {
            append_to_history("user", user_prompt);
        
            // Format prompt and tokenize
            std::string full_prompt = format_chat_template();
            std::vector<int32_t> current_tokens = model.tokenizer->Encode(full_prompt);
        
            std::print("Assistant: ");
            std::fflush(stdout);
        
            std::string assistant_reply;
        
            // Autoregressive Generation Loop
            for (size_t step = 0; step < max_new_tokens; ++step) {
                // Run forward pass through the engine
                int32_t next_token = inference_engine.prefill(current_tokens);
        
                // Check for EOS
                if (is_stopping_token(next_token)) {
                    break;
                }
        
                // Stream decode directly to the terminal!
                std::string piece = model.tokenizer->Decode({next_token});
                std::print("{}", piece);
                std::fflush(stdout);
        
                assistant_reply += piece;
                current_tokens.push_back(next_token);
            }
            std::println("");
        
            // Save reply to history for multi-turn context
            append_to_history("assistant", assistant_reply);
            return assistant_reply;
        }
        
        
};
