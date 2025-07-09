package com.example.demo.service;

import com.example.demo.entity.RecommendLog;
import com.example.demo.repository.RecommendLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class RecommendLogService {

    private final RecommendLogRepository recommendLogRepository;

    public void saveLog(String nickname, String gender, String ageGroup, int residence, Long productId) {
        RecommendLog log = RecommendLog.builder()
                .nickname(nickname)
                .gender(gender)
                .ageGroup(ageGroup)
                .residence(residence)
                .productId(productId)
                .timestamp(LocalDateTime.now())
                .build();

        recommendLogRepository.save(log);
    }


}
