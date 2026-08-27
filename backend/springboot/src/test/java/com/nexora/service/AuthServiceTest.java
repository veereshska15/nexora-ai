package com.nexora.service;

import com.nexora.config.JwtConfig;
import com.nexora.dto.request.RegisterRequest;
import com.nexora.dto.response.AuthResponse;
import com.nexora.entity.Role;
import com.nexora.entity.RoleName;
import com.nexora.entity.User;
import com.nexora.exception.AuthException;
import com.nexora.repository.RoleRepository;
import com.nexora.repository.UserRepository;
import com.nexora.security.JwtService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AuthServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private RoleRepository roleRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @Mock
    private JwtService jwtService;

    @Mock
    private SessionService sessionService;

    @Mock
    private AuditService auditService;

    @Mock
    private JwtConfig jwtConfig;

    @InjectMocks
    private AuthService authService;

    private RegisterRequest registerRequest;
    private User mockUser;
    private Role defaultRole;

    @BeforeEach
    void setUp() {
        registerRequest = RegisterRequest.builder()
                .email("student@nexora.ai")
                .password("SecurePass123!")
                .fullName("Student User")
                .build();

        defaultRole = Role.builder()
                .id(UUID.randomUUID())
                .name(RoleName.ROLE_USER)
                .description("Standard user")
                .build();

        mockUser = User.builder()
                .id(UUID.randomUUID())
                .email("student@nexora.ai")
                .passwordHash("hashedPass")
                .fullName("Student User")
                .role("ROLE_USER")
                .build();
    }

    @Test
    void register_Success() {
        when(userRepository.existsByEmail("student@nexora.ai")).thenReturn(false);
        when(roleRepository.findByName(RoleName.ROLE_USER)).thenReturn(Optional.of(defaultRole));
        when(passwordEncoder.encode(any())).thenReturn("hashedPass");
        when(userRepository.save(any())).thenReturn(mockUser);
        when(jwtService.generateAccessTokenForUser(any(), any(), any())).thenReturn("mockAccessToken");
        when(jwtService.generateRefreshToken(any(), any())).thenReturn("mockRefreshToken");
        when(jwtConfig.getExpirationMs()).thenReturn(86400000L);

        AuthResponse response = authService.register(registerRequest, "127.0.0.1", "req-1");

        assertNotNull(response);
        assertEquals("student@nexora.ai", response.getEmail());
        assertEquals("mockAccessToken", response.getAccessToken());
        verify(userRepository, times(1)).save(any());
        verify(auditService, times(1)).logAuditEvent(any(), eq("USER_REGISTERED"), any(), any(), any());
    }

    @Test
    void register_DuplicateEmail_ThrowsAuthException() {
        when(userRepository.existsByEmail("student@nexora.ai")).thenReturn(true);

        assertThrows(AuthException.class, () -> authService.register(registerRequest, "127.0.0.1", "req-2"));
        verify(userRepository, never()).save(any());
    }
}
