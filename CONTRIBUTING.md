
Contributing to Persephone web API
==================================

The preferred workflow for contributing to Persephone web API is to fork the
main repository on GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/persephone-tools/persephone-web-API)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account.

2. Clone your fork of the Persephone web API repository from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:persephone-tools/persephone-web-API.git
   $ cd persephone
   ```

3. Create a ``feature`` or ``bugfix`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a branch. It's good practice to never work on the ``master`` branch!
   You don't need to worry about the history, changes can be rebased or squashed as needed.

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Go to the GitHub web page of your fork of the Persephone-web-API repo. Click the
  'Pull request' button to send your changes to the project's maintainers for
  review. This will send an email to the maintainers.

(If you are unfamiliar with Git, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

Please follow the following rules before you submit a pull request:

- If your pull request addresses an issue, please use the pull request title
  to describe the issue and mention the issue number in the pull request description.
  This will make sure a link back to the original issue is created.

- All public functions and methods should ideally have informative docstrings.

- Please prefix the title of your pull request with `[MRG]` (Ready for
  Merge), if the contribution is complete and ready for a detailed review.
  Incomplete contributions should be prefixed `[WIP]` (to indicate a work
  in progress) and changed to `[MRG]` when it matures. WIPs may be useful
  to: indicate you are working on something to avoid duplicated work,
  request broad review of functionality or API, or seek collaborators.
  WIPs often benefit from the inclusion of a
  [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
  in the pull request description.

- Documentation and coverage tests are very beneficial for enhancements to be accepted.

- Any changes to the API must be consistent with the API specification document. If a
  change creates a conflict between the API specification and the API behavior in production
  it is unlikely to be accepted. Please make sure any API changes are valid as per the API
  specification by updating the API specification document if necessary.

You can also check for common programming errors with the following tools:

- Code lint, check with:

```bash
pip install pylint
pylint path/to/code
```

Pylint errors are indicative of critical issues, if your changes introduce Pylint errors they are unlikely to be accepted.

Filing bugs
-----------

We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

- Verify that your issue is not being currently addressed by other
  [issues](https://github.com/persephone-tools/persephone-web-API/issues?q=)
  or [pull requests](https://github.com/persephone-tools/persephone-web-API/pulls?q=).

- Please ensure all code snippets and error messages are formatted in
  appropriate code blocks.
  See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

- If the bug is system specific please include your operating system type and version number,
  as well as your Python versions. This information can be found by running the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  ```

- Issues relating to the API not accepting or producing the correct data types are greatly aided by including the exact data that was involved.
  By providing this you will make it much easier for the developers to fix the bug.

Questions about contributing
----------------------------

Please feel free to make a post on the [discussion emailing list](https://lists.persephone-asr.org/postorius/lists/discuss.lists.persephone-asr.org/) 
with any questions you have regarding contributions. Core contributors check that list regularly and would be happy to answer any questions you have about contributing or getting involved with the project.