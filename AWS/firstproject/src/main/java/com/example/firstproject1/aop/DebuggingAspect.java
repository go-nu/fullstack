package com.example.firstproject1.aop;

import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

@Aspect // AOP 클래스 선언: 부가 기능을 주입하는 클래스
@Component// Ioc 컨테이너가 해당 객체를 생성 및 관리
@Slf4j
public class DebuggingAspect {
    @Pointcut("execution(* com.example.firstproject1.service.CommentService.*(..))")
    private void cut(){}

    @Before("cut()")
    public void loggingArgs(JoinPoint joinPoint){
        // 입력값 가져오기
        Object[] args = joinPoint.getArgs();

        // 클래스명 - commentService
        String className = joinPoint.getTarget().getClass().getSimpleName();

        // 메소드명 - create
        String methodName = joinPoint.getSignature().getName();

        for(Object obj : args) { // forEach
            log.info("{}#{}의 입력값 => {}", className,methodName,obj);
        }


    }
@AfterReturning(value = "cut()", returning = "returnObj")
public void loggingReturnValue(JoinPoint joinPoint, Object returnObj) {
    // 클래스명
    String className = joinPoint.getTarget().getClass().getSimpleName();

    // 메소드명
    String methodName = joinPoint.getSignature().getName();

    // 반환값 로깅
    // CommentService#create()의 반환값 => CommentDto(id=10,...)
    log.info("{}#{}의 반환값 => {}", className,methodName,returnObj);
}

}
