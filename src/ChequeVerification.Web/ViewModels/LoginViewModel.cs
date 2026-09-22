using System.ComponentModel.DataAnnotations;

namespace ChequeVerification.Web.ViewModels;

public class LoginViewModel
{
    [Required(ErrorMessage = "L'email est requis.")]
    [EmailAddress(ErrorMessage = "Adresse email invalide.")]
    [Display(Name = "Email")]
    public string Email { get; set; } = string.Empty;

    [Required(ErrorMessage = "Le mot de passe est requis.")]
    [DataType(DataType.Password)]
    [Display(Name = "Mot de passe")]
    public string Password { get; set; } = string.Empty;

    [Display(Name = "Se souvenir de moi")]
    public bool RememberMe { get; set; }

    public string? ReturnUrl { get; set; }
}
