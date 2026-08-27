package com.nexora.controller;

import com.nexora.dto.response.ApiResponse;
import com.nexora.dto.response.UserResponse;
import com.nexora.security.UserPrincipal;
import com.nexora.service.UserService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/me")
    public ResponseEntity<ApiResponse<UserResponse>> getCurrentUser(
            @AuthenticationPrincipal UserPrincipal currentUser,
            HttpServletRequest servletRequest) {

        String requestId = servletRequest.getHeader("X-Request-ID");
        UserResponse userResponse = userService.getUserProfile(currentUser.getId());
        return ResponseEntity.ok(ApiResponse.success(userResponse, "Current user profile fetched", requestId != null ? requestId : "me"));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<UserResponse>> getUserById(
            @PathVariable UUID id,
            HttpServletRequest servletRequest) {

        String requestId = servletRequest.getHeader("X-Request-ID");
        UserResponse userResponse = userService.getUserProfile(id);
        return ResponseEntity.ok(ApiResponse.success(userResponse, "User profile fetched", requestId != null ? requestId : "user-id"));
    }
}
