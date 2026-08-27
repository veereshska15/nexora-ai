package com.nexora.controller;

import com.nexora.dto.request.LoginRequest;
import com.nexora.dto.request.RefreshTokenRequest;
import com.nexora.dto.request.RegisterRequest;
import com.nexora.dto.response.ApiResponse;
import com.nexora.dto.response.AuthResponse;
import com.nexora.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    public ResponseEntity<ApiResponse<AuthResponse>> register(
            @Valid @RequestBody RegisterRequest request,
            HttpServletRequest servletRequest) {

        String ipAddress = servletRequest.getRemoteAddr();
        String requestId = servletRequest.getHeader("X-Request-ID");

        AuthResponse authResponse = authService.register(request, ipAddress, requestId != null ? requestId : "reg");
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(authResponse, "User registered successfully", requestId != null ? requestId : "reg"));
    }

    @PostMapping("/login")
    public ResponseEntity<ApiResponse<AuthResponse>> login(
            @Valid @RequestBody LoginRequest request,
            HttpServletRequest servletRequest) {

        String ipAddress = servletRequest.getRemoteAddr();
        String requestId = servletRequest.getHeader("X-Request-ID");

        AuthResponse authResponse = authService.login(request, ipAddress, requestId != null ? requestId : "login");
        return ResponseEntity.ok(ApiResponse.success(authResponse, "Login successful", requestId != null ? requestId : "login"));
    }

    @PostMapping("/refresh")
    public ResponseEntity<ApiResponse<AuthResponse>> refreshToken(
            @Valid @RequestBody RefreshTokenRequest request,
            HttpServletRequest servletRequest) {

        String ipAddress = servletRequest.getRemoteAddr();
        String requestId = servletRequest.getHeader("X-Request-ID");

        AuthResponse authResponse = authService.refreshToken(request, ipAddress, requestId != null ? requestId : "refresh");
        return ResponseEntity.ok(ApiResponse.success(authResponse, "Token refreshed successfully", requestId != null ? requestId : "refresh"));
    }

    @PostMapping("/logout")
    public ResponseEntity<ApiResponse<String>> logout(HttpServletRequest servletRequest) {
        String requestId = servletRequest.getHeader("X-Request-ID");
        return ResponseEntity.ok(ApiResponse.success("Logged out successfully", "Logout successful", requestId != null ? requestId : "logout"));
    }
}
