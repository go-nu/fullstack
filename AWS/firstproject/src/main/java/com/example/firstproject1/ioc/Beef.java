package com.example.firstproject1.ioc;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter

public class Beef extends Ingredient{
    public Beef(String name){
        super(name);
    }
}
