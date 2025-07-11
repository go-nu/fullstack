package com.example.demo.constant;

public enum ProductModelSelect {
    // SMALL, MEDIUM, LARGE, DEFAULT_MODEL

    SUPER_SINGLE("슈퍼싱글"),
    QUEEN("퀸"),
    KING("킹"),
    DEFAULT_MODEL("기본");

    private final String label;

    ProductModelSelect(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }

}
