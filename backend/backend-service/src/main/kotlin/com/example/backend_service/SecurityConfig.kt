package com.example.backend_service

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.web.SecurityFilterChain

@Configuration
@EnableWebSecurity
class SecurityConfig {

    @Bean
    fun securityFilterChain(http: HttpSecurity): SecurityFilterChain {
        http
                .csrf { it.disable() } // Disable CSRF; re-enable in production as needed
                .authorizeRequests {
                    it.antMatchers("/checkImageQuality", "/test").permitAll() // Allow access to this path
                    it.anyRequest().authenticated() // Require authentication for other requests
                }

        return http.build()
    }
}
