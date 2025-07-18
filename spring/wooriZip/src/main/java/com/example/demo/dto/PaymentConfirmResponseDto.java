package com.example.demo.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true) // 예기치 않은 필드 무시
public class PaymentConfirmResponseDto {
    private String method;
    private String message;
    private String code;
    private EasyPay easyPay;

    public static PaymentConfirmResponseDto fromJson(String json) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(json, PaymentConfirmResponseDto.class);
    }

    public String getEasyPayProvider() {
        return easyPay != null ? easyPay.getProvider() : "N/A";
    }

    @Getter
    @Setter
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class EasyPay {
        private String provider;
    }
}
