package com.nexora.controller;

import com.nexora.dto.request.LoginRequest;
import com.nexora.dto.request.RegisterRequest;
import com.nexora.dto.response.AuthResponse;
import com.nexora.service.AuthService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;

import org.springframework.mock.web.MockHttpServletRequest;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AuthControllerTest {

    @Mock
    private AuthService authService;

    @InjectMocks
    private AuthController authController;

    private MockHttpServletRequest request;
    private AuthResponse mockAuthResponse;

    @BeforeEach
    void setUp() {
        request = new MockHttpServletRequest();
        request.setRemoteAddr("127.0.0.1");
        request.addHeader("X-Request-ID", "test-req-id");

        mockAuthResponse = AuthResponse.builder()
                .userId(UUID.randomUUID())
                .email("user@nexora.ai")
                .fullName("Test User")
                .role("ROLE_USER")
                .accessToken("mockAccessToken")
                .refreshToken("mockRefreshToken")
                .build();
    }

    @Test
    void register_Returns201Created() {
        RegisterRequest registerRequest = RegisterRequest.builder()
                .email("user@nexora.ai")
                .password("Password123!")
                .fullName("Test User")
                .build();

        when(authService.register(any(), any(), any())).thenReturn(mockAuthResponse);

        var response = authController.register(registerRequest, request);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals("user@nexora.ai", response.getBody().getData().getEmail());
    }

    @Test
    void login_Returns200OK() {
        LoginRequest loginRequest = LoginRequest.builder()
                .email("user@nexora.ai")
                .password("Password123!")
                .build();

        when(authService.login(any(), any(), any())).thenReturn(mockAuthResponse);

        var response = authController.login(loginRequest, request);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals("mockAccessToken", response.getBody().getData().getAccessToken());
    }
}
