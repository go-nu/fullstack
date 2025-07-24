package com.example.firstproject1.ioc;

public abstract class Ingredient {
    private String name;
    public Ingredient(String name) {
        this.name = name;
    }

    public Ingredient() {
    }

    public String getName() {
        return name;
    }


}
