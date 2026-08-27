package com.nexora.service;

import com.nexora.config.JwtConfig;
import com.nexora.dto.request.LoginRequest;
import com.nexora.dto.request.RefreshTokenRequest;
import com.nexora.dto.request.RegisterRequest;
import com.nexora.dto.response.AuthResponse;
import com.nexora.entity.Role;
import com.nexora.entity.RoleName;
import com.nexora.entity.User;
import com.nexora.exception.AuthException;
import com.nexora.repository.RoleRepository;
import com.nexora.repository.UserRepository;
import com.nexora.security.JwtService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashSet;
import java.util.Set;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final RoleRepository roleRepository;
    private final PasswordEncoder passwordEncoder;
    private final AuthenticationManager authenticationManager;
    private final JwtService jwtService;
    private final SessionService sessionService;
    private final AuditService auditService;
    private final JwtConfig jwtConfig;

    @Transactional
    public AuthResponse register(RegisterRequest request, String ipAddress, String requestId) {
        if (userRepository.existsByEmail(request.getEmail())) {
            auditService.logSecurityEvent(null, "REGISTRATION_FAILED", "MEDIUM", "Email already exists: " + request.getEmail(), ipAddress, requestId);
            throw new AuthException("Email is already registered: " + request.getEmail());
        }

        Role defaultRole = roleRepository.findByName(RoleName.ROLE_USER)
                .orElseGet(() -> roleRepository.save(Role.builder().name(RoleName.ROLE_USER).description("Standard registered user").build()));

        Set<Role> roles = new HashSet<>();
        roles.add(defaultRole);

        User user = User.builder()
                .email(request.getEmail().toLowerCase())
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .fullName(request.getFullName())
                .role("ROLE_USER")
                .isActive(true)
                .roles(roles)
                .build();

        User savedUser = userRepository.save(user);

        String accessToken = jwtService.generateAccessTokenForUser(savedUser.getId(), savedUser.getEmail(), savedUser.getRole());
        String refreshToken = jwtService.generateRefreshToken(savedUser.getId(), savedUser.getEmail());

        sessionService.createSession(savedUser, refreshToken, ipAddress, "NEXORA-Flutter-Client");
        auditService.logAuditEvent(savedUser.getId(), "USER_REGISTERED", "Registered new user: " + savedUser.getEmail(), ipAddress, requestId);

        return AuthResponse.builder()
                .userId(savedUser.getId())
                .email(savedUser.getEmail())
                .fullName(savedUser.getFullName())
                .role(savedUser.getRole())
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .expiresInMs(jwtConfig.getExpirationMs())
                .build();
    }

    @Transactional
    public AuthResponse login(LoginRequest request, String ipAddress, String requestId) {
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(request.getEmail().toLowerCase(), request.getPassword())
            );

            User user = userRepository.findByEmail(request.getEmail().toLowerCase())
                    .orElseThrow(() -> new AuthException("User not found"));

            String accessToken = jwtService.generateAccessToken(authentication);
            String refreshToken = jwtService.generateRefreshToken(user.getId(), user.getEmail());

            sessionService.createSession(user, refreshToken, ipAddress, "NEXORA-Flutter-Client");
            auditService.logAuditEvent(user.getId(), "USER_LOGIN", "User logged in successfully", ipAddress, requestId);

            return AuthResponse.builder()
                    .userId(user.getId())
                    .email(user.getEmail())
                    .fullName(user.getFullName())
                    .role(user.getRole())
                    .accessToken(accessToken)
                    .refreshToken(refreshToken)
                    .tokenType("Bearer")
                    .expiresInMs(jwtConfig.getExpirationMs())
                    .build();
        } catch (Exception e) {
            auditService.logSecurityEvent(null, "LOGIN_FAILED", "MEDIUM", "Failed login for email: " + request.getEmail(), ipAddress, requestId);
            throw new AuthException("Invalid email or password");
        }
    }

    @Transactional
    public AuthResponse refreshToken(RefreshTokenRequest request, String ipAddress, String requestId) {
        String token = request.getRefreshToken();
        if (jwtService.isTokenExpired(token)) {
            throw new AuthException("Refresh token is expired");
        }

        String email = jwtService.getUsernameFromToken(token);
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new AuthException("User not found"));

        String newAccessToken = jwtService.generateAccessTokenForUser(user.getId(), user.getEmail(), user.getRole());
        String newRefreshToken = jwtService.generateRefreshToken(user.getId(), user.getEmail());

        sessionService.createSession(user, newRefreshToken, ipAddress, "NEXORA-Flutter-Client");
        auditService.logAuditEvent(user.getId(), "TOKEN_REFRESH", "Refreshed access token", ipAddress, requestId);

        return AuthResponse.builder()
                .userId(user.getId())
                .email(user.getEmail())
                .fullName(user.getFullName())
                .role(user.getRole())
                .accessToken(newAccessToken)
                .refreshToken(newRefreshToken)
                .tokenType("Bearer")
                .expiresInMs(jwtConfig.getExpirationMs())
                .build();
    }
}
