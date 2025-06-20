package board.kkw.service;

import board.kkw.domain.Member;
import board.kkw.dto.MemberDto;
import board.kkw.repository.MemberRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MemberService {
    @Autowired
    private MemberRepository memberRepository;

    public Member signUp(MemberDto dto) {
        Member member = new Member();
        member.setUserId(dto.getUserId());
        member.setPassword(dto.getPassword());
        return memberRepository.save(member);
    }

    public Member login(MemberDto dto) {
        return memberRepository.findByUserId(dto.getUserId())
                .filter(m -> m.getPassword().equals(dto.getPassword()))
                .orElse(null);
    }
}
