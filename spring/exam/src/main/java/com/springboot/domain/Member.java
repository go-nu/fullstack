package com.springboot.domain;

import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class Member {

    @MemberId
    private String memberId;

    @Size(min = 4, max = 10, message = "4자~10자 이내로 입력")
    private String passwd;
}
