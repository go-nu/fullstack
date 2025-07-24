package com.example.firstproject1.ioc;

import org.springframework.stereotype.Component;

@Component
public class Chef {
    // 셰프는 식재료 공장을 알고있어야하기 때문에 변수를 가지고있어야함.
    private IngredientFactory ingredientFactory;

    // 셰프가 식재료 공장과 협업하기 위한 DI
    // 외부에서 공장(객체)에 대한 정보를 받아옴.
    public Chef(IngredientFactory ingredientFactory) {
        this.ingredientFactory = ingredientFactory;
    }
    public String cook(String menu){

        Ingredient ingredient = ingredientFactory.get(menu);
        // 요리 반환
        return ingredient.getName() + "으로 만든 " + menu;
    }



}
