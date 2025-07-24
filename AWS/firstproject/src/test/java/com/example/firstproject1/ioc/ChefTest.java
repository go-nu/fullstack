package com.example.firstproject1.ioc;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class ChefTest {
//    @Autowired
//    IngredientFactory ingredientFactory;

    @Autowired
    Chef chef;
    @Test
    void 돈까스_요리하기(){
        // 준비
        String menu = "돈가스";
        // 수행
        String food = chef.cook(menu);
        // 예상
        String expected = "한돈 등심으로 만든 돈가스";
        // 검증
        assertEquals(expected, food);
        System.out.println(food);
    }

    @Test
    void 스테이크_요리하기(){
        // 준비
        String menu = "스테이크";
        // 수행
        String food = chef.cook(menu);
        // 예상
        String expected = "한우 꽃등심으로 만든 스테이크";
        // 검증
        assertEquals(expected, food);
        System.out.println(food);

    }
    @Test
    void 크리스피_치킨_요리하기(){
        // 준비
        String menu = "크리스피 치킨";
        // 수행
        String food = chef.cook(menu);
        // 예상
        String expected = "국내산 10호 닭으로 만든 크리스피 치킨";
        // 검증
        assertEquals(expected, food);
        System.out.println(food);

    }
}

//IngredientFactory 클래스는 @Component 스프링빈으로 등록
// 이클래스는 메뉴에 따라 적절한  Ingredient 객체생성하여 반환

//chef  클래스는 @Component 스프링빈으로 등록 , 생성자를 통해
//IngredientFactory 객체 주입 이를 통해 Chef 클래스는
// IngredientFactory 협업해서 요리 준비
