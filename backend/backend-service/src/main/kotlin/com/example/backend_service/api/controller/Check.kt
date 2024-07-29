package com.example.backend_service.api.controller

import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class Check {

    @GetMapping("/test")
    fun testEndpoint(): String {
        return "The service is running"
    }
}