// eslint-disable-next-line n/no-unpublished-import -- This import is not published cos it's dev only
import defaultConfig from '@hughescr/eslint-config-default';
// eslint-disable-next-line n/no-unpublished-import -- This import is not published cos it's dev only
import tseslint from 'typescript-eslint';

export default
[
    {
        name: 'ignores',
        ignores: ['coverage', 'node_modules', 'venv'],
    },
    defaultConfig.configs.recommended,
    ...tseslint.configs.recommended,
    ...tseslint.configs.stylistic,
    {
        rules: {
            '@stylistic/space-before-blocks': ['warn', { functions: 'always', keywords: 'always', classes: 'always' }],
            '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
        },
    },
    {
        files: ['**/*.js', '**/*.mjs'],
        ...tseslint.configs.disableTypeChecked,
    },
];
