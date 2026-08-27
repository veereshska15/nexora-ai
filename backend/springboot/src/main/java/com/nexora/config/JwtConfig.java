package com.nexora.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
@Getter
@Setter
public class JwtConfig {

    @Value("${nexora.security.jwt.secret}")
    private String secret;

    @Value("${nexora.security.jwt.expiration-ms}")
    private Long expirationMs;

    @Value("${nexora.security.jwt.refresh-expiration-ms}")
    private Long refreshExpirationMs;
}
