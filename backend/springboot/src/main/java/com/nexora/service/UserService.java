package com.nexora.service;

import com.nexora.dto.response.UserResponse;
import com.nexora.entity.Permission;
import com.nexora.entity.Role;
import com.nexora.entity.RoleName;
import com.nexora.entity.User;
import com.nexora.exception.ResourceNotFoundException;
import com.nexora.repository.RoleRepository;
import com.nexora.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final RoleRepository roleRepository;

    @Transactional(readOnly = true)
    public UserResponse getUserProfile(UUID userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + userId));

        Set<String> permissions = new HashSet<>();
        user.getRoles().forEach(r -> {
            for (Permission p : r.getPermissions()) {
                permissions.add(p.getName());
            }
        });

        return UserResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .fullName(user.getFullName())
                .role(user.getRole())
                .isActive(user.getIsActive())
                .permissions(permissions)
                .createdAt(user.getCreatedAt())
                .build();
    }

    @Transactional(readOnly = true)
    public List<UserResponse> getAllUsers() {
        return userRepository.findAll().stream()
                .map(u -> getUserProfile(u.getId()))
                .collect(Collectors.toList());
    }

    @Transactional
    public UserResponse updateUserRole(UUID userId, RoleName newRoleName) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + userId));

        Role newRole = roleRepository.findByName(newRoleName)
                .orElseGet(() -> roleRepository.save(Role.builder().name(newRoleName).description(newRoleName.name()).build()));

        user.getRoles().clear();
        user.getRoles().add(newRole);
        user.setRole(newRoleName.name());

        User updatedUser = userRepository.save(user);
        return getUserProfile(updatedUser.getId());
    }
}
