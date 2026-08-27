package com.nexora.controller;

import com.nexora.dto.response.ApiResponse;
import com.nexora.dto.response.UserResponse;
import com.nexora.entity.RoleName;
import com.nexora.service.UserService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/admin")
@RequiredArgsConstructor
public class AdminController {

    private final UserService userService;

    @GetMapping("/users")
    public ResponseEntity<ApiResponse<List<UserResponse>>> getAllUsers(HttpServletRequest servletRequest) {
        String requestId = servletRequest.getHeader("X-Request-ID");
        List<UserResponse> users = userService.getAllUsers();
        return ResponseEntity.ok(ApiResponse.success(users, "All system users fetched", requestId != null ? requestId : "admin-users"));
    }

    @PutMapping("/users/{id}/role")
    public ResponseEntity<ApiResponse<UserResponse>> updateUserRole(
            @PathVariable UUID id,
            @RequestParam RoleName roleName,
            HttpServletRequest servletRequest) {

        String requestId = servletRequest.getHeader("X-Request-ID");
        UserResponse updatedUser = userService.updateUserRole(id, roleName);
        return ResponseEntity.ok(ApiResponse.success(updatedUser, "User role updated to " + roleName, requestId != null ? requestId : "role-update"));
    }
}
