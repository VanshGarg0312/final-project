package com.fakenews.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@RestController
@RequestMapping("/news")
@CrossOrigin("*")
public class NewsController {

    @PostMapping("/predict")
    public String predict(@RequestBody Map<String, String> body) {

        RestTemplate restTemplate = new RestTemplate();

        String url = "http://localhost:8000/predict";

        return restTemplate.postForObject(url, body, String.class);
    }
}