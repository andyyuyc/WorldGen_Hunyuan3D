using UnityEngine;

/// <summary>
/// Simple first-person controller for traversing WorldGen scenes.
/// Attach to a capsule with a Camera as child.
/// </summary>
[RequireComponent(typeof(CharacterController))]
public class PlayerController : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 5.0f;
    public float sprintMultiplier = 2.0f;
    public float jumpHeight = 1.2f;
    public float gravity = -20.0f;

    [Header("Look")]
    public float mouseSensitivity = 2.0f;
    public float maxLookAngle = 85.0f;

    [Header("References")]
    public Transform cameraTransform;

    private CharacterController _controller;
    private Vector3 _velocity;
    private float _xRotation;
    private bool _cursorLocked = true;

    void Start()
    {
        _controller = GetComponent<CharacterController>();

        if (cameraTransform == null)
        {
            Camera cam = GetComponentInChildren<Camera>();
            if (cam != null)
                cameraTransform = cam.transform;
        }

        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        HandleCursorLock();
        if (_cursorLocked)
        {
            HandleLook();
        }
        HandleMovement();
    }

    void HandleLook()
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

        _xRotation -= mouseY;
        _xRotation = Mathf.Clamp(_xRotation, -maxLookAngle, maxLookAngle);

        if (cameraTransform != null)
        {
            cameraTransform.localRotation = Quaternion.Euler(_xRotation, 0, 0);
        }
        transform.Rotate(Vector3.up * mouseX);
    }

    void HandleMovement()
    {
        // Ground check
        bool isGrounded = _controller.isGrounded;
        if (isGrounded && _velocity.y < 0)
        {
            _velocity.y = -2f;
        }

        // Input
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");

        Vector3 move = transform.right * h + transform.forward * v;

        // Sprint
        float speed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift))
        {
            speed *= sprintMultiplier;
        }

        _controller.Move(move * speed * Time.deltaTime);

        // Jump
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            _velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }

        // Gravity
        _velocity.y += gravity * Time.deltaTime;
        _controller.Move(_velocity * Time.deltaTime);
    }

    void HandleCursorLock()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            _cursorLocked = !_cursorLocked;
            Cursor.lockState = _cursorLocked ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !_cursorLocked;
        }
    }
}
