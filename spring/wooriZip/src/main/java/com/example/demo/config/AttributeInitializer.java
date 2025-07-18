package com.example.demo.config;

import com.example.demo.entity.Attribute;
import com.example.demo.entity.AttributeValue;
import com.example.demo.repository.AttributeRepository;
import com.example.demo.repository.AttributeValueRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class AttributeInitializer implements CommandLineRunner {
    @Autowired
    private AttributeRepository attributeRepository;

    @Autowired
    private AttributeValueRepository attributeValueRepository;

    @Override
    public void run(String... args) {
        // 사이즈
        Attribute size = attributeRepository.findByName("사이즈")
                .orElseGet(() -> attributeRepository.save(new Attribute("사이즈")));
        addAttributeValueIfNotExists(size, "S");
        addAttributeValueIfNotExists(size, "M");
        addAttributeValueIfNotExists(size, "L");

        // 색상
        Attribute color = attributeRepository.findByName("색상")
                .orElseGet(() -> attributeRepository.save(new Attribute("색상")));
        addAttributeValueIfNotExists(color, "블랙");
        addAttributeValueIfNotExists(color, "화이트");

        // 소재
        Attribute material = attributeRepository.findByName("소재")
                .orElseGet(() -> attributeRepository.save(new Attribute("소재")));
        addAttributeValueIfNotExists(material, "원목");
    }

    private void addAttributeValueIfNotExists(Attribute attribute, String value) {
        if (!attributeValueRepository.existsByAttributeAndValue(attribute, value)) {
            attributeValueRepository.save(new AttributeValue(attribute, value));
        }
    }
}