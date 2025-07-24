package com.example.demo.controller;

import com.example.demo.dto.ChatbotResponseDto;
import com.example.demo.service.ChatbotService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Controller
@RequiredArgsConstructor
public class ChatbotController {

    private final ChatbotService chatbotService;

    @GetMapping("/chatbot")
    public String chatbotPage() {
        return "chatbot/chatbot";
    }

    @PostMapping("/api/chatbot/chat")
    @ResponseBody
    public ResponseEntity<ChatbotResponseDto> chat(@RequestParam String message) {
        ChatbotResponseDto response = chatbotService.processMessage(message);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/api/chatbot/suggestions")
    @ResponseBody
    public ResponseEntity<List<String>> getSuggestions() {
        List<String> suggestions = chatbotService.getCommonSuggestions();
        return ResponseEntity.ok(suggestions);
    }
} 