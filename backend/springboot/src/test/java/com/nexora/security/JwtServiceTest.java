package com.nexora.security;

import com.nexora.config.JwtConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class JwtServiceTest {

    @Mock
    private JwtConfig jwtConfig;

    @InjectMocks
    private JwtService jwtService;

    private final String secret = "nexora-super-secret-jwt-signing-key-change-in-production-min-32-chars!";

    @BeforeEach
    void setUp() {
        lenient().when(jwtConfig.getSecret()).thenReturn(secret);
        lenient().when(jwtConfig.getExpirationMs()).thenReturn(3600000L); // 1 hour
        lenient().when(jwtConfig.getRefreshExpirationMs()).thenReturn(86400000L);
    }

    @Test
    void generateAccessToken_and_Validate() {
        UUID userId = UUID.randomUUID();
        String email = "admin@nexora.ai";
        String role = "ROLE_ADMIN";

        String token = jwtService.generateAccessTokenForUser(userId, email, role);

        assertNotNull(token);
        assertEquals(email, jwtService.getUsernameFromToken(token));
        assertTrue(jwtService.validateToken(token, email));
        assertFalse(jwtService.isTokenExpired(token));
    }
}
